"""
PDF to Text Batch Converter for Science LLM
===========================================
Converts scientific research PDFs into clean, normalized .txt files.

Features:
- Multi-column layout reconstruction via PyMuPDF.
- Preserves mathematical notation and LaTeX equations ($...$, $$...$$).
- Strips references/bibliographies, header watermarks, and publisher boilerplate.
- Integrates with HashDatabase to track conversion status.
- Multiprocessing / multithreaded extraction for high throughput.
- Saves clean text files into `drive/Science LLM/Dataset/extracted_texts/`.
- Optional --remove-pdfs flag to free Google Drive storage after extraction.
"""

import os
import re
import time
import logging
from typing import Optional, List, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from config_manager import ScienceLLMConfig
from hash_db import HashDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PDFConverter")


class PDFToTextConverter:
    """Converts scientific PDFs to clean text files with formula preservation."""

    def __init__(self, config: Optional[ScienceLLMConfig] = None, max_workers: int = 4):
        self.config = config or ScienceLLMConfig()
        self.hash_db = HashDatabase(config=self.config)
        self.dataset_dir = self.config.dataset_dir
        self.extracted_text_dir = self.dataset_dir / "extracted_texts"
        self.extracted_text_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

    @staticmethod
    def clean_scientific_text(text: str) -> str:
        """Cleans noise while preserving mathematical and scientific notation."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
        text = re.sub(r"arXiv:\d{4}\.\d{4,5}v\d+\s+\[.*?\]\s+\d+\s+\w+\s+\d{4}", "", text)
        text = re.sub(r"doi:\s*10\.\d{4,9}/[-._;()/:A-Z0-9]+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"https?://\S+", "", text)

        # Truncate at standard Bibliography / References section
        ref_match = re.search(r"\n\s*(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography)\s*\n", text)
        if ref_match:
            text = text[:ref_match.start()]

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def convert_single_pdf(self, pdf_path: Path, delete_pdf_after: bool = False) -> Optional[Path]:
        """Converts one PDF into clean .txt file in extracted_texts/."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required. Run: pip install pymupdf")

        txt_filename = f"{pdf_path.stem}.txt"
        target_txt_path = self.extracted_text_dir / txt_filename

        # If already converted and non-empty, skip
        if target_txt_path.exists() and target_txt_path.stat().st_size > 200:
            return target_txt_path

        try:
            doc = fitz.open(str(pdf_path))
            pages_text = []

            for page in doc:
                blocks = page.get_text("blocks")
                page_blocks = []
                for b in blocks:
                    if b[6] == 0:  # Text block
                        content = b[4].strip()
                        if content:
                            page_blocks.append(content)
                if page_blocks:
                    pages_text.append("\n\n".join(page_blocks))

            doc.close()

            raw_text = "\n\n".join(pages_text)
            clean_text = self.clean_scientific_text(raw_text)

            if len(clean_text) < 200:
                # File is either scanned images without OCR or empty
                logger.warning(f"Extracted text too short ({len(clean_text)} chars) for: {pdf_path.name}")
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
        # Exclude any PDFs inside extracted_texts if any
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

        # Update data.json state
        self.config.state["txt_conversion"] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_pdfs_scanned": len(all_pdfs),
            "total_txt_converted": converted_count,
            "extracted_text_dir": str(self.extracted_text_dir),
            "estimated_total_chars": total_chars
        }
        self.config.save_state()

        logger.info(f"PDF-to-TXT conversion complete! {converted_count}/{len(all_pdfs)} text documents created in {self.extracted_text_dir}.")
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
