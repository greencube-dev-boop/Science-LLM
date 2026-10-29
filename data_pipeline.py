"""
Data Engineering Pipeline for Science Domain LLM
================================================
Handles:
1. Ingesting converted scientific texts from `Dataset/extracted_texts/`.
2. Generator-based low-memory sequence packing into 2048-token blocks.
3. Document boundary demarcation using model-specific EOS / end-of-text tokens.
4. Synchronizing dataset statistics and token metrics into `data.json`.
"""

import os
import re
import json
import logging
from typing import List, Dict, Optional, Generator
from pathlib import Path

from config_manager import ScienceLLMConfig
from downloader import PDFDownloader
from hash_db import HashDatabase
from pdf_converter import PDFToTextConverter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DataPipeline")


def download_pdfs_from_list(config: Optional[ScienceLLMConfig] = None, down_file: Optional[str] = None) -> List[str]:
    """Helper to check and download PDFs from down.txt to Dataset folder."""
    cfg = config or ScienceLLMConfig()
    dl = PDFDownloader(config=cfg)
    return dl.process_down_txt(down_txt_path=down_file)


class DatasetBuilder:
    """Processes scientific text into tokenized sequences ready for Continued Pre-Training."""

    def __init__(self, tokenizer_name: str = "Qwen/Qwen2.5-3B", max_seq_length: int = 2048, config: Optional[ScienceLLMConfig] = None):
        from transformers import AutoTokenizer
        self.config = config or ScienceLLMConfig()
        self.max_seq_length = max_seq_length
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def process_dataset_folder(self, input_dir: Optional[str] = None, output_jsonl: Optional[str] = None) -> int:
        """
        Reads all converted .txt, .md, and remaining .pdf files in Dataset directory,
        prioritizing fast pre-converted text files from extracted_texts/.
        """
        input_path = Path(input_dir) if input_dir else self.config.dataset_dir
        out_file = Path(output_jsonl) if output_jsonl else self.config.dataset_dir / "processed_science_corpus.jsonl"
        records_count = 0
        total_chars = 0

        logger.info(f"Scanning for pre-converted text files and PDFs in: {input_path}")
        with open(out_file, "w", encoding="utf-8") as out_f:
            # Check extracted_texts first for fast text loading
            txt_files = list(input_path.glob("extracted_texts/*.txt")) + [
                f for f in input_path.glob("*.txt") if "extracted_texts" not in str(f)
            ]
            for file_path in txt_files:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as in_f:
                        text = in_f.read().strip()
                        if len(text) > 200:
                            out_f.write(json.dumps({"text": text, "source": file_path.name}) + "\n")
                            records_count += 1
                            total_chars += len(text)
                except Exception as e:
                    logger.warning(f"Could not read {file_path.name}: {e}")

            # If no extracted txt files, fallback to direct PDF parsing
            if records_count == 0:
                converter = PDFToTextConverter(config=self.config)
                for file_path in input_path.glob("*.pdf"):
                    try:
                        extracted = converter.convert_single_pdf(file_path)
                        if extracted and extracted.exists():
                            with open(extracted, "r", encoding="utf-8", errors="ignore") as in_f:
                                text = in_f.read().strip()
                                if len(text) > 200:
                                    out_f.write(json.dumps({"text": text, "source": file_path.name}) + "\n")
                                    records_count += 1
                                    total_chars += len(text)
                    except Exception as e:
                        logger.warning(f"Failed to process {file_path.name}: {e}")

        logger.info(f"Corpus compilation complete: {records_count} documents ({total_chars:,} characters) saved to {out_file}")
        return records_count

    def pack_sequences_for_dapt(self, input_jsonl: Optional[str] = None, output_tokenized_path: Optional[str] = None):
        """
        Packs corpus into fixed-length token blocks with boundary markers (<|endoftext|>).
        Uses streaming batch generation to minimize RAM footprint.
        """
        from datasets import Dataset

        in_file = Path(input_jsonl) if input_jsonl else self.config.dataset_dir / "processed_science_corpus.jsonl"
        out_dir = Path(output_tokenized_path) if output_tokenized_path else self.config.dataset_dir / "tokenized_science_blocks"

        if not in_file.exists():
            logger.warning(f"Processed corpus not found at {in_file}")
            return None

        eos_id = self.tokenizer.eos_token_id
        seq_len = self.max_seq_length

        def token_block_generator() -> Generator[Dict[str, List[int]], None, None]:
            buffer: List[int] = []
            with open(in_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        tokens = self.tokenizer.encode(text, add_special_tokens=False)
                        tokens.append(eos_id)
                        buffer.extend(tokens)

                        while len(buffer) >= seq_len:
                            block = buffer[:seq_len]
                            buffer = buffer[seq_len:]
                            yield {"input_ids": block, "labels": list(block)}

        logger.info("Packing sequences via generator into token blocks...")
        dataset = Dataset.from_generator(token_block_generator)
        dataset.save_to_disk(str(out_dir))

        total_tokens = len(dataset) * seq_len
        logger.info(f"Saved packed dataset ({len(dataset):,} blocks = ~{total_tokens:,} tokens) to {out_dir}")

        self.config.state["dataset_stats"] = {
            "total_tokens": total_tokens,
            "packed_blocks": len(dataset),
            "block_size": seq_len,
            "dataset_path": str(out_dir)
        }
        self.config.save_state()
        return dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Science Data Pipeline")
    parser.add_argument("--base_dir", type=str, default=None, help="Base Drive dir: drive/Science LLM")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B", help="Tokenizer model name")
    parser.add_argument("--download_first", action="store_true", help="Download links from down.txt before processing")
    parser.add_argument("--convert_pdfs", action="store_true", help="Batch convert PDFs to clean TXT before building corpus")
    args = parser.parse_args()

    cfg = ScienceLLMConfig(base_dir=args.base_dir)

    if args.download_first or (cfg.base_dir / "down.txt").exists():
        download_pdfs_from_list(config=cfg)

    if args.convert_pdfs:
        conv = PDFToTextConverter(config=cfg)
        conv.convert_all()

    builder = DatasetBuilder(tokenizer_name=args.model_name, max_seq_length=2048, config=cfg)
    records = builder.process_dataset_folder()

    if records > 0:
        builder.pack_sequences_for_dapt()
    else:
        logger.info(f"No documents found in {cfg.dataset_dir}.")
