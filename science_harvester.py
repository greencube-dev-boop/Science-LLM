"""
Science Harvester: Open Scientific Paper & Book Downloader by Domain Tags
========================================================================
Searches and downloads open-access research papers and books across scientific tags.

Default Tags:
- organic chemistry
- biochemistry
- botany
- plant biology
- chemistry
- biology

Sources:
- Europe PMC Open Access API (Biochemistry, Botany, Plant Biology, Molecular Chemistry)
- arXiv Open Access API (Chemical Physics, Quantitative Biology, Organic Chemistry)

Features:
- SHA-256 Hash Database integration for zero-duplicate storage.
- Validates PDF structure (%PDF header).
- Polite rate-limiting and multithreaded downloading.
- Configurable targets: 1,000 to 5,000+ publications.
"""

import os
import re
import time
import json
import logging
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from config_manager import ScienceLLMConfig
from hash_db import HashDatabase

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


class ScienceHarvester:
    """Harvests open-access papers by scientific tags into Dataset/."""

    def __init__(
        self,
        config: Optional[ScienceLLMConfig] = None,
        tags: Optional[List[str]] = None,
        max_workers: int = 4
    ):
        self.config = config or ScienceLLMConfig()
        self.tags = tags or DEFAULT_SCIENCE_TAGS
        self.hash_db = HashDatabase(config=self.config)
        self.dataset_dir = self.config.dataset_dir
        self.max_workers = max_workers

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
                # Atom namespace
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                entries = root.findall("atom:entry", ns)
                if not entries:
                    break

                for entry in entries:
                    title_elem = entry.find("atom:title", ns)
                    title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else "paper"

                    pdf_url = None
                    for link in entry.findall("atom:link", ns):
                        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                            pdf_url = link.attrib.get("href")
                            break

                    if not pdf_url:
                        # Fallback to arxiv id link
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

                time.sleep(2.0)  # Courtesy delay for arXiv API
            except Exception as e:
                logger.warning(f"Error querying arXiv for tag '{tag}': {e}")
                break

        logger.info(f"[arXiv] Found {len(publications)} candidate papers for tag '{tag}'.")
        return publications

    def search_europe_pmc(self, tag: str, max_results: int = 500) -> List[Dict[str, str]]:
        """Queries Europe PMC for open-access peer-reviewed papers with full-text PDFs."""
        logger.info(f"[EuropePMC] Searching for tag: '{tag}' (Limit: {max_results})...")
        encoded_query = urllib.parse.quote(f'("{tag}") AND (OPEN_ACCESS:Y) AND (HAS_PDF:Y)')
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
                    pmcid = item.get("pmcid")
                    doi = item.get("doi")

                    pdf_url = None
                    # Search full-text URL list for direct PDF
                    for link_entry in item.get("fullTextUrlList", {}).get("fullTextUrl", []):
                        if link_entry.get("documentStyle") == "pdf":
                            pdf_url = link_entry.get("url")
                            break

                    if not pdf_url and pmcid:
                        # Direct PMC open access PDF link
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

    def download_paper(self, paper: Dict[str, str]) -> Optional[Path]:
        """Downloads a single publication, validates %PDF header, and registers in Hash DB."""
        url = paper["url"]
        title = paper.get("title", "paper")
        tag = paper.get("tag", "general")

        if self.hash_db.is_url_seen(url):
            return None

        clean_t = self.clean_filename(title)
        safe_tag = self.clean_filename(tag)
        filename = f"{safe_tag}_{clean_t}_{int(time.time()*1000)%100000}.pdf"
        target_path = self.dataset_dir / filename
        temp_path = self.dataset_dir / f"{filename}.part"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/pdf"
        }

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=40) as response:
                with open(temp_path, "wb") as f:
                    while chunk := response.read(65536):
                        f.write(chunk)

            # Validate %PDF header
            if not temp_path.exists() or temp_path.stat().st_size < 1000:
                temp_path.unlink(missing_ok=True)
                return None

            with open(temp_path, "rb") as f:
                header = f.read(1024)
                if b"%PDF" not in header:
                    temp_path.unlink(missing_ok=True)
                    return None

            # Move to target path
            temp_path.rename(target_path)

            # Register in SHA-256 database (removes duplicates automatically)
            is_unique = self.hash_db.register_file(
                file_path=target_path,
                source_url=url,
                title=title,
                tag=tag,
                metadata={"source": paper.get("source")}
            )

            if is_unique:
                mb_size = target_path.stat().st_size / (1024 * 1024)
                logger.info(f"[Downloaded] {filename} ({mb_size:.2f} MB) [{tag}]")
                return target_path
            return None

        except Exception as e:
            temp_path.unlink(missing_ok=True)
            return None

    def harvest(self, total_target: int = 1000) -> int:
        """
        Coordinates discovery and multithreaded download across all tags
        until total_target is reached.
        """
        per_tag_target = max(50, total_target // len(self.tags))
        logger.info(f"Initiating Science Harvester: Target = {total_target} papers across {len(self.tags)} tags (~{per_tag_target} per tag).")

        all_candidates: List[Dict[str, str]] = []

        for tag in self.tags:
            # Discover from Europe PMC (dominant in biology, chemistry, botany)
            pmc_papers = self.search_europe_pmc(tag, max_results=per_tag_target)
            all_candidates.extend(pmc_papers)

            # Discover from arXiv (dominant in chemistry & molecular biology)
            arxiv_papers = self.search_arxiv(tag, max_results=max(50, per_tag_target // 2))
            all_candidates.extend(arxiv_papers)

        logger.info(f"Total candidate publications discovered: {len(all_candidates):,}. Starting multithreaded download...")

        downloaded_count = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_paper = {
                executor.submit(self.download_paper, paper): paper
                for paper in all_candidates
            }

            for future in as_completed(future_to_paper):
                res = future.result()
                if res is not None:
                    downloaded_count += 1
                    if downloaded_count % 25 == 0:
                        logger.info(f"Progress: {downloaded_count}/{total_target} new unique papers downloaded.")
                    if downloaded_count >= total_target:
                        logger.info(f"Reached target quota of {total_target} papers!")
                        break

        # Log to data.json
        self.config.state["harvester_runs"] = self.config.state.get("harvester_runs", [])
        self.config.state["harvester_runs"].append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target": total_target,
            "downloaded": downloaded_count,
            "tags": self.tags,
            "total_unique_db_files": self.hash_db.data.get("total_unique_files", 0)
        })
        self.config.save_state()

        logger.info(f"Harvest complete! Successfully saved {downloaded_count} new scientific publications into {self.dataset_dir}.")
        return downloaded_count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Science Harvester for Open Papers & Books")
    parser.add_argument("--base_dir", type=str, default=None, help="Base Drive directory")
    parser.add_argument("--limit", type=int, default=1000, help="Target number of publications (1000 - 5000)")
    parser.add_argument("--workers", type=int, default=4, help="Download threads")
    parser.add_argument("--tags", nargs="+", default=DEFAULT_SCIENCE_TAGS, help="List of science tags")
    args = parser.parse_args()

    cfg = ScienceLLMConfig(base_dir=args.base_dir)
    harvester = ScienceHarvester(config=cfg, tags=args.tags, max_workers=args.workers)
    harvester.harvest(total_target=args.limit)
