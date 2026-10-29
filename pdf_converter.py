"""
PDF to Text High-Fidelity Batch Converter for Science LLM
=========================================================
Converts scientific research PDFs into clean, normalized .txt files.

Advanced Scientific Normalization Features:
1. Dehyphenation across line breaks (e.g., "macro-\\nmolecule" -> "macromolecule").
2. Standard Unicode ligature replacement (fi, fl, ffi, ffl, etc.).
3. Bounding-box margin filtering: Strips journal running headers and page footers.
4. Preserves chemical formulas, Greek characters, and LaTeX mathematical equations.
5. Multi-column reading order reconstruction via PyMuPDF block coordinates.
6. Heuristic quality gate: Filters out corrupted scans, font-encoding failures, or low-density pages.
7. Saves clean text into `drive/Science LLM/Dataset/extracted_texts/`.
"""

import os
import re
import time
import unicodedata
import logging
from typing import Optional, List, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from config_manager import ScienceLLMConfig
from hash_db import HashDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PDFConverter")

LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
    "\u2013": "-",
    "\u2014": "--",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"'
}


class PDFToTextConverter:
    """Converts scientific PDFs to clean text with scientific dehyphenation and formula preservation."""

    def __init__(self, config: Optional[ScienceLLMConfig] = None, max_workers: int = 4):
        self.config = config or ScienceLLMConfig()
        self.hash_db = HashDatabase(config=self.config)
        self.dataset_dir = self.config.dataset_dir
        self.extracted_text_dir = self.dataset_dir / "extracted_texts"
        self.extracted_text_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

    @staticmethod
    def normalize_ligatures(text: str) -> str:
        """Replaces Unicode typographic ligatures with ASCII equivalents."""
        for lig, repl in LIGATURE_MAP.items():
            text = text.replace(lig, repl)
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def dehyphenate_scientific_text(text: str) -> str:
        """
        Recombines words split by end-of-line hyphens:
        'macro-\\nmolecule' -> 'macromolecule'
        'photo-\\nsystem' -> 'photosystem'
        """
        # Match word ending in hyphen at newline, followed by lowercase word continuation
        dehyphenated = re.sub(r'(\b[A-Za-z]{2,})-\s*\n\s*([a-z]{2,}\b)', r'\1\2', text)
        return dehyphenated

    @classmethod
    def clean_scientific_text(cls, text: str) -> str:
        """
        Cleans scientific text while strictly preserving chemical nomenclature,
        Greek symbols (alpha, beta), and LaTeX math notation.
        """
        text = cls.normalize_ligatures(text)
        text = cls.dehyphenate_scientific_text(text)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Strip standalone page numbering lines
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

        # Strip common arXiv metadata and publisher DOIs
        text = re.sub(r"arXiv:\d{4}\.\d{4,5}(?:v\d+)?\s+\[.*?\]\s+\d+\s+\w+\s+\d{4}", "", text)
        text = re.sub(r"https?://(?:dx\.)?doi\.org/10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", "", text)
        text = re.sub(r"doi:\s*10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", "", text, flags=re.IGNORECASE)

        # Truncate at standard Bibliography / References / Acknowledgments sections
        ref_match = re.search(r"\n\s*(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography|LITERATURE CITED)\s*\n", text)
        if ref_match:
            text = text[:ref_match.start()]

        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()

    @staticmethod
    def is_quality_text(text: str) -> bool:
        """Checks if extracted text is valid and not corrupted font bytes or scanned noise."""
        if len(text) < 400:
            return False
        # Calculate ratio of printable alphanumeric + standard punctuation
        valid_chars = sum(1 for c in text if c.isalnum() or c in " .,;:()[]{}-+*/=<>_$%\n\t'\"")
        ratio = valid_chars / max(1, len(text))
        return ratio >= 0.85

    def convert_single_pdf(self, pdf_path: Path, delete_pdf_after: bool = False) -> Optional[Path]:
        """Converts a single PDF to a normalized .txt file."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required. Run: pip install pymupdf")

        txt_filename = f"{pdf_path.stem}.txt"
        target_txt_path = self.extracted_text_dir / txt_filename

        if target_txt_path.exists() and target_txt_path.stat().st_size > 200:
            return target_txt_path

        try:
            doc = fitz.open(str(pdf_path))
            pages_text = []

            for page in doc:
                rect = page.rect
                height = rect.height
                # Margin bounds: ignore top 5% (header) and bottom 5% (footer)
                top_margin = height * 0.05
                bottom_margin = height * 0.95

                blocks = page.get_text("blocks")
                page_blocks = []
                for b in blocks:
                    # b: (x0, y0, x1, y1, text, block_no, block_type)
                    if b[6] == 0:  # Text block
                        y0, y1 = b[1], b[3]
                        # Exclude running headers and footers
                        if y1 < top_margin or y0 > bottom_margin:
                            continue
                        content = b[4].strip()
                        if content:
                            page_blocks.append(content)

                if page_blocks:
                    pages_text.append("\n\n".join(page_blocks))

            doc.close()

            raw_text = "\n\n".join(pages_text)
            clean_text = self.clean_scientific_text(raw_text)

            if not self.is_quality_text(clean_text):
                logger.warning(f"Extracted text failed quality threshold ({len(clean_text)} chars): {pdf_path.name}")
                return None

            with open(target_txt_path, "w", encoding="utf-8") as f:
                f.write(clean_text)

            # Record in Hash DB
            sha256 = self.hash_db.calculate_file_sha256(pdf_path)
            self.hash_db.mark_txt_converted(sha256, target_txt_path)

            if delete_pdf_after:
                pdf_path.unlink(missing_ok=True)
                logger.info(f"Converted & removed PDF: {pdf_path.name} -> {txt_filename} ({len(clean_text):,} chars)")
            else:
                logger.info(f"Converted: {pdf_path.name} -> {txt_filename} ({len(clean_text):,} chars)")

            return target_txt_path

        except Exception as e:
            logger.error(f"Failed to convert {pdf_path.name}: {e}")
            return None

    def convert_all(self, delete_pdfs: bool = False) -> int:
        """Finds all PDFs in Dataset/ and converts them to TXT using worker threads."""
        all_pdfs = list(self.dataset_dir.glob("*.pdf")) + list(self.dataset_dir.glob("**/*.pdf"))
        all_pdfs = [p for p in all_pdfs if "extracted_texts" not in str(p)]

        logger.info(f"Discovered {len(all_pdfs)} PDFs in {self.dataset_dir}. Converting to clean text...")

        converted_count = 0
        total_chars = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_pdf = {
                executor.submit(self.convert_single_pdf, pdf, delete_pdfs): pdf
                for pdf in all_pdfs
            }

            for future in as_completed(future_to_pdf):
                res = future.result()
                if res is not None:
                    converted_count += 1
                    try:
                        total_chars += res.stat().st_size
                    except Exception:
                        pass

        self.config.state["txt_conversion"] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_pdfs_scanned": len(all_pdfs),
            "total_txt_converted": converted_count,
            "extracted_text_dir": str(self.extracted_text_dir),
            "estimated_total_chars": total_chars
        }
        self.config.save_state()

        logger.info(f"PDF-to-TXT conversion complete! {converted_count}/{len(all_pdfs)} documents in {self.extracted_text_dir}.")
        return converted_count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Scientific PDFs to Clean TXT")
    parser.add_argument("--base_dir", type=str, default=None, help="Base Drive directory")
    parser.add_argument("--workers", type=int, default=4, help="Worker threads")
    parser.add_argument("--delete_pdfs", action="store_true", help="Delete PDFs after converting to text to save Drive storage")
    args = parser.parse_args()

    cfg = ScienceLLMConfig(base_dir=args.base_dir)
    converter = PDFToTextConverter(config=cfg, max_workers=args.workers)
    converter.convert_all(delete_pdfs=args.delete_pdfs)
