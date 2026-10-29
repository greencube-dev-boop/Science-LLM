"""
Hash Database & Deduplication Manager for Science LLM
======================================================
Maintains a persistent SHA-256 hash database in `drive/Science LLM/config/hash_db.json`
to guarantee zero duplicate downloads across runs and mirrors.
"""

import os
import json
import time
import shutil
import hashlib
import logging
from typing import Dict, Any, Optional, Set
from pathlib import Path

from config_manager import ScienceLLMConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("HashDB")


class HashDatabase:
    """Tracks SHA-256 content hashes and URL history to eliminate duplicate files."""

    def __init__(self, config: Optional[ScienceLLMConfig] = None):
        self.config = config or ScienceLLMConfig()
        self.db_path = self.config.config_dir / "hash_db.json"
        self.data: Dict[str, Any] = self._load_or_init_db()

    def _load_or_init_db(self) -> Dict[str, Any]:
        """Loads existing hash database or creates an empty one."""
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    db = json.load(f)
                    logger.info(f"Loaded Hash Database: {len(db.get('hashes', {}))} unique hashes recorded.")
                    return db
            except Exception as e:
                logger.warning(f"Error reading hash_db.json: {e}. Reinitializing backup.")
                shutil.copyfile(self.db_path, self.config.config_dir / f"hash_db.json.bak.{int(time.time())}")

        initial_db = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_unique_files": 0,
            "hashes": {},       # sha256 -> metadata
            "seen_urls": []     # list of processed URLs
        }
        self.save_db(initial_db)
        return initial_db

    def save_db(self, data: Optional[Dict[str, Any]] = None):
        """Atomically saves hash database to disk."""
        if data is not None:
            self.data = data
        self.data["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.data["total_unique_files"] = len(self.data.get("hashes", {}))

        tmp_file = self.config.config_dir / f"hash_db.tmp.{os.getpid()}"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        shutil.move(str(tmp_file), str(self.db_path))

    @staticmethod
    def calculate_file_sha256(file_path: Path) -> str:
        """Computes SHA-256 checksum of a file in 64KB chunks."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def is_url_seen(self, url: str) -> bool:
        """Checks if a URL has already been queued or downloaded."""
        return url in self.data.get("seen_urls", [])

    def is_hash_known(self, sha256: str) -> bool:
        """Checks if content hash is already in the database."""
        return sha256 in self.data.get("hashes", {})

    def register_file(
        self,
        file_path: Path,
        source_url: Optional[str] = None,
        title: Optional[str] = None,
        tag: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Calculates SHA-256 for file_path. If hash already exists, deletes duplicate
        and returns False. Otherwise registers and returns True.
        """
        if not file_path.exists():
            return False

        file_sha256 = self.calculate_file_sha256(file_path)

        if self.is_hash_known(file_sha256):
            existing_info = self.data["hashes"][file_sha256]
            logger.info(f"Duplicate detected: {file_path.name} has identical SHA-256 as {existing_info.get('filename')}. Removing duplicate.")
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove duplicate file: {e}")
            return False

        # Register unique file
        self.data["hashes"][file_sha256] = {
            "filename": file_path.name,
            "path": str(file_path),
            "size_bytes": file_path.stat().st_size,
            "source_url": source_url,
            "title": title or file_path.stem,
            "tag": tag or "general_science",
            "txt_converted": False,
            "txt_path": None,
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "extra": metadata or {}
        }

        if source_url and source_url not in self.data["seen_urls"]:
            self.data["seen_urls"].append(source_url)

        self.save_db()
        return True

    def mark_txt_converted(self, file_sha256: str, txt_path: Path):
        """Updates record when PDF has been extracted to TXT."""
        if file_sha256 in self.data["hashes"]:
            self.data["hashes"][file_sha256]["txt_converted"] = True
            self.data["hashes"][file_sha256]["txt_path"] = str(txt_path)
            self.save_db()


if __name__ == "__main__":
    db = HashDatabase()
    print(f"Hash Database ready: {db.db_path}")
    print(f"Total Unique Files: {db.data['total_unique_files']}")
