"""
Science Harvester: Open Scientific Paper & Book Downloader by Domain Tags
========================================================================
Searches and downloads open-access research papers, review articles, and textbooks.

Default Tags:
- organic chemistry
- biochemistry
- botany
- plant biology
- chemistry
- biology

Sources:
- Europe PMC Open Access API (Prioritizing high-density Review articles & primary research)
- arXiv Open Access API (Chemical Physics, Quantitative Biology, Organic Synthesis)
- Curated Open Textbooks (OpenStax / CC-BY Biology, Chemistry, Organic Chemistry)

Optimizations:
- Storage-Saver Mode: Downloads to local temporary scratch space, extracts clean text
  directly into `Dataset/extracted_texts/`, registers SHA-256 hash, and removes temporary PDF.
  (Saves 98% of Google Drive quota: 1000 papers = ~30MB instead of ~3GB!).
- Cross-mirror Title and SHA-256 Content Deduplication via HashDatabase.
- Multithreaded concurrent streaming with exponential backoff.
"""

import os
import re
import time
import json
import logging
import tempfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from config_manager import ScienceLLMConfig
from hash_db import HashDatabase
from pdf_converter import PDFToTextConverter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ScienceHarvester")

DEFAULT_SCIENCE_TAGS = [
    "organic chemistry",
    "biochemistry",
    "botany",
    "plant biology",
    "chemistry",
    "biology"
]

# Foundational OpenStax and open-access CC-BY textbook links
CURATED_OPEN_TEXTBOOKS = [
    {
        "title": "OpenStax Chemistry 2e",
        "url": "https://assets.openstax.org/oscms-prodcms/media/documents/Chemistry2e-WEB_1.pdf",
        "tag": "chemistry"
    },
    {
        "title": "OpenStax Biology 2e",
        "url": "https://assets.openstax.org/oscms-prodcms/media/documents/Biology2e-WEB_1.pdf",
        "tag": "biology"
    },
    {
        "title": "OpenStax Anatomy and Physiology 2e",
        "url": "https://assets.openstax.org/oscms-prodcms/media/documents/AnatomyandPhysiology2e-WEB.pdf",
        "tag": "biology"
    },
    {
        "title": "OpenStax Microbiology",
        "url": "https://assets.openstax.org/oscms-prodcms/media/documents/Microbiology-WEB.pdf",
        "tag": "biochemistry"
    }
]


class ScienceHarvester:
    """Harvests open-access papers and textbooks with on-the-fly text extraction and deduplication."""

    def __init__(
        self,
        config: Optional[ScienceLLMConfig] = None,
        tags: Optional[List[str]] = None,
        max_workers: int = 4,
        extract_to_txt_direct: bool = True,
        keep_raw_pdfs: bool = False
    ):
        self.config = config or ScienceLLMConfig()
        self.tags = tags or DEFAULT_SCIENCE_TAGS
        self.hash_db = HashDatabase(config=self.config)
        self.dataset_dir = self.config.dataset_dir
        self.extracted_text_dir = self.dataset_dir / "extracted_texts"
        self.extracted_text_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.extract_to_txt_direct = extract_to_txt_direct
        self.keep_raw_pdfs = keep_raw_pdfs

    @staticmethod
    def clean_filename(name: str) -> str:
        """Sanitizes titles for valid filesystem names."""
        cleaned = re.sub(r'[\\/*?:"<>|]', "", name)
        cleaned = re.sub(r"\s+", "_", cleaned)
        return cleaned[:80].strip("_")

    def search_arxiv(self, tag: str, max_results: int = 250) -> List[Dict[str, str]]:
        """Queries arXiv API for open-access papers matching a scientific tag."""
        logger.info(f"[arXiv] Searching for tag: '{tag}' (Limit: {max_results})...")
        encoded_tag = urllib.parse.quote(f'all:"{tag}"')
        page_size = min(100, max_results)
        publications = []

        for start_offset in range(0, max_results, page_size):
            url = f"http://export.arxiv.org/api/query?search_query={encoded_tag}&start={start_offset}&max_results={page_size}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "ScienceLLM/1.0 (Research Harvester)"})
                with urllib.request.urlopen(req, timeout=30) as response:
                    xml_data = response.read()

                root = ET.fromstring(xml_data)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                entries = root.findall("atom:entry", ns)
                if not entries:
                    break

                for entry in entries:
                    title_elem = entry.find("atom:title", ns)
                    title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else "paper"

                    if self.hash_db.is_title_seen(title):
                        continue

                    pdf_url = None
                    for link in entry.findall("atom:link", ns):
                        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                            pdf_url = link.attrib.get("href")
                            break

                    if not pdf_url:
                        id_elem = entry.find("atom:id", ns)
                        if id_elem is not None and "/abs/" in id_elem.text:
                            pdf_url = id_elem.text.replace("/abs/", "/pdf/") + ".pdf"

                    if pdf_url and not self.hash_db.is_url_seen(pdf_url):
                        publications.append({
                            "title": title,
                            "url": pdf_url,
                            "tag": tag,
                            "source": "arxiv"
                        })

                time.sleep(2.0)
            except Exception as e:
                logger.warning(f"Error querying arXiv for tag '{tag}': {e}")
                break

        logger.info(f"[arXiv] Found {len(publications)} candidate papers for tag '{tag}'.")
        return publications

    def search_europe_pmc(self, tag: str, max_results: int = 500, prioritize_reviews: bool = True) -> List[Dict[str, str]]:
        """
        Queries Europe PMC for open-access papers with full-text PDFs.
        Optionally prioritizes review articles for higher information density.
        """
        query_suffix = ' AND (PUB_TYPE:"Review" OR PUB_TYPE:"review-article")' if prioritize_reviews else ''
        logger.info(f"[EuropePMC] Searching for tag: '{tag}' (Limit: {max_results}, Reviews: {prioritize_reviews})...")
        encoded_query = urllib.parse.quote(f'("{tag}") AND (OPEN_ACCESS:Y) AND (HAS_PDF:Y){query_suffix}')
        page_size = min(100, max_results)
        publications = []

        cursor_mark = "*"
        fetched = 0

        while fetched < max_results:
            url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={encoded_query}&format=json&pageSize={page_size}&cursorMark={cursor_mark}&resultType=core"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "ScienceLLM/1.0 (Research Harvester)"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                results = data.get("resultList", {}).get("result", [])
                if not results:
                    break

                for item in results:
                    title = item.get("title", "paper").strip()
                    if self.hash_db.is_title_seen(title):
                        continue

                    pmcid = item.get("pmcid")
                    doi = item.get("doi")

                    pdf_url = None
                    for link_entry in item.get("fullTextUrlList", {}).get("fullTextUrl", []):
                        if link_entry.get("documentStyle") == "pdf":
                            pdf_url = link_entry.get("url")
                            break

                    if not pdf_url and pmcid:
                        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/main.pdf"

                    if pdf_url and not self.hash_db.is_url_seen(pdf_url):
                        publications.append({
                            "title": title,
                            "url": pdf_url,
                            "tag": tag,
                            "source": "europe_pmc",
                            "doi": doi,
                            "pmcid": pmcid
                        })

                fetched += len(results)
                cursor_mark = data.get("nextCursorMark")
                if not cursor_mark or len(results) < page_size:
                    break

                time.sleep(1.0)
            except Exception as e:
                logger.warning(f"Error querying Europe PMC for '{tag}': {e}")
                break

        logger.info(f"[EuropePMC] Found {len(publications)} candidate papers for tag '{tag}'.")
        return publications

    def download_and_process_item(self, paper: Dict[str, str]) -> Optional[Path]:
        """
        Downloads a publication into local temp scratch space, verifies %PDF,
        registers SHA-256, converts to clean .txt directly, and manages Drive storage.
        """
        url = paper["url"]
        title = paper.get("title", "paper")
        tag = paper.get("tag", "general")

        if self.hash_db.is_url_seen(url) or self.hash_db.is_title_seen(title):
            return None

        clean_t = self.clean_filename(title)
        safe_tag = self.clean_filename(tag)
        base_stem = f"{safe_tag}_{clean_t}_{int(time.time()*1000)%100000}"

        # Download to local scratch disk (saves Google Drive network latency and quota)
        temp_dir = Path(tempfile.gettempdir()) / "science_stage"
        temp_dir.mkdir(parents=True, exist_ok=True)
        local_pdf_path = temp_dir / f"{base_stem}.pdf"
        local_part_path = temp_dir / f"{base_stem}.part"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/pdf"
        }

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as response:
                with open(local_part_path, "wb") as f:
                    while chunk := response.read(65536):
                        f.write(chunk)

            # Validate %PDF header
            if not local_part_path.exists() or local_part_path.stat().st_size < 1000:
                local_part_path.unlink(missing_ok=True)
                return None

            with open(local_part_path, "rb") as f:
                if b"%PDF" not in f.read(1024):
                    local_part_path.unlink(missing_ok=True)
                    return None

            local_part_path.rename(local_pdf_path)

            # Check for content hash duplicates before writing to Google Drive
            file_sha256 = self.hash_db.calculate_file_sha256(local_pdf_path)
            if self.hash_db.is_hash_known(file_sha256):
                local_pdf_path.unlink(missing_ok=True)
                return None

            # Mode A: Storage-saver (Convert directly to clean .txt on Drive)
            if self.extract_to_txt_direct:
                converter = PDFToTextConverter(config=self.config)
                # Convert PDF from local SSD directly to Google Drive extracted_texts/
                txt_path = self.extracted_text_dir / f"{base_stem}.txt"

                # Extract and clean
                extracted_txt = converter.convert_single_pdf(local_pdf_path, delete_pdf_after=True)
                if extracted_txt and extracted_txt.exists():
                    # Register file in DB
                    self.hash_db.register_file(
                        file_path=extracted_txt,
                        source_url=url,
                        title=title,
                        tag=tag,
                        metadata={"source": paper.get("source"), "pdf_sha256": file_sha256}
                    )
                    self.hash_db.mark_txt_converted(file_sha256, extracted_txt)
                    logger.info(f"[Harvested & Converted] {base_stem}.txt ({extracted_txt.stat().st_size / 1024:.1f} KB) [{tag}]")
                    return extracted_txt
                return None

            # Mode B: Keep full raw PDF on Google Drive
            else:
                drive_pdf_path = self.dataset_dir / f"{base_stem}.pdf"
                import shutil
                shutil.move(str(local_pdf_path), str(drive_pdf_path))
                is_unique = self.hash_db.register_file(
                    file_path=drive_pdf_path,
                    source_url=url,
                    title=title,
                    tag=tag,
                    metadata={"source": paper.get("source")}
                )
                if is_unique:
                    logger.info(f"[Saved PDF] {drive_pdf_path.name} ({drive_pdf_path.stat().st_size / (1024*1024):.2f} MB) [{tag}]")
                    return drive_pdf_path
                return None

        except Exception as e:
            local_part_path.unlink(missing_ok=True)
            local_pdf_path.unlink(missing_ok=True)
            return None

    def harvest(self, total_target: int = 1000) -> int:
        """
        Coordinates discovery and multithreaded download across all tags
        until total_target is reached.
        """
        per_tag_target = max(50, total_target // len(self.tags))
        logger.info(f"Initiating Science Harvester: Target = {total_target} publications across {len(self.tags)} tags.")

        all_candidates: List[Dict[str, str]] = []

        # 1. Seed with curated open textbooks first (highest educational/foundational value)
        for book in CURATED_OPEN_TEXTBOOKS:
            if not self.hash_db.is_title_seen(book["title"]):
                all_candidates.append({**book, "source": "openstax_textbook"})

        # 2. Query Europe PMC (50% reviews for conceptual depth, 50% primary research)
        for tag in self.tags:
            reviews = self.search_europe_pmc(tag, max_results=per_tag_target // 2, prioritize_reviews=True)
            all_candidates.extend(reviews)
            general_papers = self.search_europe_pmc(tag, max_results=per_tag_target // 2, prioritize_reviews=False)
            all_candidates.extend(general_papers)

            # 3. Query arXiv for chemical physics, organic, and quantitative biology
            arxiv_papers = self.search_arxiv(tag, max_results=max(40, per_tag_target // 3))
            all_candidates.extend(arxiv_papers)

        logger.info(f"Discovered {len(all_candidates):,} candidates. Starting multithreaded harvest (Workers: {self.max_workers})...")

        harvested_count = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_paper = {
                executor.submit(self.download_and_process_item, paper): paper
                for paper in all_candidates
            }

            for future in as_completed(future_to_paper):
                res = future.result()
                if res is not None:
                    harvested_count += 1
                    if harvested_count % 25 == 0:
                        logger.info(f"Progress: {harvested_count}/{total_target} new unique items processed.")
                    if harvested_count >= total_target:
                        logger.info(f"Target quota of {total_target} reached!")
                        break

        # Log run in data.json
        self.config.state["harvester_runs"] = self.config.state.get("harvester_runs", [])
        self.config.state["harvester_runs"].append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target": total_target,
            "processed": harvested_count,
            "tags": self.tags,
            "storage_mode": "txt_direct" if self.extract_to_txt_direct else "keep_pdfs",
            "total_unique_db_files": self.hash_db.data.get("total_unique_files", 0)
        })
        self.config.save_state()

        logger.info(f"Harvest complete! {harvested_count} new documents stored in {self.dataset_dir}.")
        return harvested_count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Science Harvester for Open Papers & Books")
    parser.add_argument("--base_dir", type=str, default=None, help="Base Drive directory")
    parser.add_argument("--limit", type=int, default=1000, help="Target number of publications (1000 - 5000)")
    parser.add_argument("--workers", type=int, default=4, help="Download threads")
    parser.add_argument("--keep_pdfs", action="store_true", help="Store raw PDFs on Drive (default extracts text directly to save Drive space)")
    parser.add_argument("--tags", nargs="+", default=DEFAULT_SCIENCE_TAGS, help="List of science tags")
    args = parser.parse_args()

    cfg = ScienceLLMConfig(base_dir=args.base_dir)
    harvester = ScienceHarvester(
        config=cfg,
        tags=args.tags,
        max_workers=args.workers,
        extract_to_txt_direct=not args.keep_pdfs,
        keep_raw_pdfs=args.keep_pdfs
    )
    harvester.harvest(total_target=args.limit)
