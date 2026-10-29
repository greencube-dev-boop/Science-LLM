"""
PDF Batch Downloader for Science LLM
====================================
Reads download URLs from `drive/Science LLM/down.txt`
and downloads PDFs directly into `drive/Science LLM/Dataset`.

Features:
- Streaming downloads with progress logging.
- Content-Disposition and URL filename extraction.
- PDF magic byte verification (%PDF) to detect invalid links or HTML paywalls.
- Skip-if-already-downloaded logic to save bandwidth on re-runs.
- Synchronizes download history into `drive/Science LLM/config/data.json`.
"""

import os
import re
import time
import hashlib
import logging
import urllib.parse
from typing import List, Dict, Optional
from pathlib import Path
import urllib.request

from config_manager import ScienceLLMConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PDFDownloader")


class PDFDownloader:
    """Manages reading down.txt and downloading PDFs into Dataset/."""

    def __init__(self, config: Optional[ScienceLLMConfig] = None):
        self.config = config or ScienceLLMConfig()
        self.down_txt_path = self.config.base_dir / "down.txt"
        self.dataset_dir = self.config.dataset_dir

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Removes illegal characters from filenames."""
        cleaned = re.sub(r'[\\/*?:"<>|]', "_", filename)
        return cleaned.strip()

    def get_filename_from_url(self, url: str) -> str:
        """Extracts a valid .pdf filename from URL or creates a deterministic hash name."""
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.strip("/")
        basename = os.path.basename(path)

        if basename and basename.lower().endswith(".pdf"):
            filename = self.sanitize_filename(basename)
        else:
            # Hash URL if no obvious PDF filename in path (e.g. arXiv /pdf/2301.12345)
            url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
            clean_stem = self.sanitize_filename(basename if basename else "doc")
            filename = f"{clean_stem}_{url_hash}.pdf"

        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"
        return filename

    def is_valid_pdf(self, file_path: Path) -> bool:
        """Verifies if the downloaded file has the %PDF header."""
        if not file_path.exists() or file_path.stat().st_size < 100:
            return False
        try:
            with open(file_path, "rb") as f:
                header = f.read(1024)
                return b"%PDF" in header
        except Exception:
            return False

    def download_url(self, url: str) -> Optional[Path]:
        """Downloads a single URL into dataset_dir with streaming and validation."""
        filename = self.get_filename_from_url(url)
        target_path = self.dataset_dir / filename

        # Skip if already exists and valid
        if target_path.exists() and self.is_valid_pdf(target_path):
            logger.info(f"File already downloaded and valid: {filename}")
            return target_path

        logger.info(f"Downloading: {url} -> {filename}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/pdf,application/xhtml+xml,text/html;q=0.9,*/*;q=0.8"
        }

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as response:
                content_type = response.headers.get("Content-Type", "")
                
                # Check Content-Disposition for server-provided filename
                content_disp = response.headers.get("Content-Disposition", "")
                if "filename=" in content_disp:
                    disp_name = re.findall(r'filename=["\']?([^"\';]+)["\']?', content_disp)
                    if disp_name:
                        filename = self.sanitize_filename(disp_name[0])
                        if not filename.lower().endswith(".pdf"):
                            filename += ".pdf"
                        target_path = self.dataset_dir / filename

                # Stream to disk
                temp_path = self.dataset_dir / f"{filename}.part"
                with open(temp_path, "wb") as out_file:
                    while True:
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        out_file.write(chunk)

                # Validate downloaded bytes
                if self.is_valid_pdf(temp_path):
                    if target_path.exists():
                        target_path.unlink()
                    temp_path.rename(target_path)
                    file_size_mb = target_path.stat().st_size / (1024 * 1024)
                    logger.info(f"Successfully saved {filename} ({file_size_mb:.2f} MB)")
                    return target_path
                else:
                    temp_path.unlink(missing_ok=True)
                    logger.warning(f"Downloaded content from {url} is not a valid PDF (likely HTML/paywall). Discarded.")
                    return None

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None

    def process_down_txt(self, down_txt_path: Optional[str] = None) -> List[str]:
        """
        Reads URLs line by line from down.txt, downloads each to Dataset/,
        and logs download records into data.json.
        """
        source_file = Path(down_txt_path) if down_txt_path else self.down_txt_path

        if not source_file.exists():
            logger.warning(f"down.txt not found at: {source_file}")
            return []

        logger.info(f"Reading download links from: {source_file}")
        with open(source_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

        logger.info(f"Found {len(lines)} links to download.")
        successful_downloads = []

        for idx, url in enumerate(lines, 1):
            logger.info(f"[{idx}/{len(lines)}] Processing link: {url}")
            saved_file = self.download_url(url)
            if saved_file:
                successful_downloads.append(str(saved_file))

        # Log into data.json state tracker
        self.config.state["download_logs"] = self.config.state.get("download_logs", [])
        self.config.state["download_logs"].append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_links_found": len(lines),
            "successfully_downloaded": len(successful_downloads),
            "down_file_source": str(source_file)
        })
        self.config.save_state()

        logger.info(f"Download batch complete! ({len(successful_downloads)}/{len(lines)} downloaded successfully into {self.dataset_dir})")
        return successful_downloads


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download PDFs from down.txt")
    parser.add_argument("--base_dir", type=str, default=None, help="Base Science LLM folder")
    parser.add_argument("--down_file", type=str, default=None, help="Path to down.txt")
    args = parser.parse_args()

    cfg = ScienceLLMConfig(base_dir=args.base_dir)
    downloader = PDFDownloader(config=cfg)
    downloader.process_down_txt(down_txt_path=args.down_file)
