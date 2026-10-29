"""
Data Engineering Pipeline for Science Domain LLM
================================================
Handles:
1. Automated PDF downloading from: drive/Science LLM/down.txt into drive/Science LLM/Dataset.
2. Ingesting scientific PDFs and text files from: drive/Science LLM/Dataset.
3. Cleaning, multi-column reassembly, and LaTeX formula preservation.
4. Packing sequences into token blocks (2048 context length).
5. Synchronizing dataset stats into drive/Science LLM/config/data.json.
"""

import os
import re
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

from config_manager import ScienceLLMConfig
from downloader import PDFDownloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DataPipeline")


def download_pdfs_from_list(config: Optional[ScienceLLMConfig] = None, down_file: Optional[str] = None) -> List[str]:
    """Helper to check and download PDFs from down.txt to Dataset folder."""
    cfg = config or ScienceLLMConfig()
    dl = PDFDownloader(config=cfg)
    return dl.process_down_txt(down_txt_path=down_file)


class ScienceTextExtractor:
    """Extracts and sanitizes text from scientific PDFs, books, and articles."""

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extracts clean text from a scientific PDF using PyMuPDF (fitz),
        respecting multi-column reading order and extracting block structures.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required for PDF extraction. Install via: pip install pymupdf")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at: {pdf_path}")

        doc = fitz.open(pdf_path)
        full_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # 'blocks' mode handles multi-column scientific layouts
            blocks = page.get_text("blocks")
            page_content = []

            for b in blocks:
                if b[6] == 0:  # Text block
                    text_block = b[4].strip()
                    if text_block:
                        page_content.append(text_block)

            full_text.append("\n\n".join(page_content))

        raw_document = "\n\n".join(full_text)
        return ScienceTextExtractor.clean_scientific_text(raw_document)

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
        Reads all .pdf, .txt, and .md files in Dataset directory, extracts & cleans them,
        and saves each document as a JSON record {'text': ...}.
        """
        input_path = Path(input_dir) if input_dir else self.config.dataset_dir
        out_file = Path(output_jsonl) if output_jsonl else self.config.dataset_dir / "processed_science_corpus.jsonl"
        records_count = 0
        total_chars = 0

        logger.info(f"Scanning for PDFs and text files in: {input_path}")
        with open(out_file, "w", encoding="utf-8") as out_f:
            for file_path in input_path.glob("**/*"):
                if file_path.suffix.lower() == ".pdf":
                    try:
                        text = ScienceTextExtractor.extract_text_from_pdf(str(file_path))
                        if len(text) > 200:
                            out_f.write(json.dumps({"text": text, "source": file_path.name}) + "\n")
                            records_count += 1
                            total_chars += len(text)
                            logger.info(f"Processed PDF: {file_path.name} ({len(text)} characters)")
                    except Exception as e:
                        logger.warning(f"Failed to process {file_path.name}: {e}")

                elif file_path.suffix.lower() in [".txt", ".md"]:
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as in_f:
                            text = in_f.read()
                            clean_txt = ScienceTextExtractor.clean_scientific_text(text)
                            if len(clean_txt) > 200:
                                out_f.write(json.dumps({"text": clean_txt, "source": file_path.name}) + "\n")
                                records_count += 1
                                total_chars += len(clean_txt)
                                logger.info(f"Processed Text: {file_path.name} ({len(clean_txt)} characters)")
                    except Exception as e:
                        logger.warning(f"Failed to process {file_path.name}: {e}")

        logger.info(f"Extraction complete. Total records: {records_count} saved to {out_file}")
        return records_count

    def pack_sequences_for_dapt(self, input_jsonl: Optional[str] = None, output_tokenized_path: Optional[str] = None):
        """Packs corpus into fixed-length token blocks and logs stats into data.json."""
        from datasets import Dataset

        in_file = Path(input_jsonl) if input_jsonl else self.config.dataset_dir / "processed_science_corpus.jsonl"
        out_dir = Path(output_tokenized_path) if output_tokenized_path else self.config.dataset_dir / "tokenized_science_blocks"

        if not in_file.exists():
            logger.warning(f"Processed corpus not found at {in_file}")
            return None

        logger.info("Packing sequences into fixed token blocks...")
        all_token_ids: List[int] = []

        with open(in_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                text = data.get("text", "")
                if text:
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    tokens.append(self.tokenizer.eos_token_id)
                    all_token_ids.extend(tokens)

        total_tokens = len(all_token_ids)
        num_blocks = total_tokens // self.max_seq_length
        logger.info(f"Total tokens: {total_tokens:,}. Creating {num_blocks:,} packed blocks of length {self.max_seq_length}.")

        packed_blocks = []
        for i in range(num_blocks):
            start_idx = i * self.max_seq_length
            end_idx = start_idx + self.max_seq_length
            block = all_token_ids[start_idx:end_idx]
            packed_blocks.append({"input_ids": block, "labels": list(block)})

        dataset = Dataset.from_list(packed_blocks)
        dataset.save_to_disk(str(out_dir))
        logger.info(f"Saved packed dataset ({len(dataset)} examples) to {out_dir}")

        self.config.state["dataset_stats"] = {
            "total_tokens": total_tokens,
            "packed_blocks": num_blocks,
            "block_size": self.max_seq_length,
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
    args = parser.parse_args()

    cfg = ScienceLLMConfig(base_dir=args.base_dir)

    if args.download_first or (cfg.base_dir / "down.txt").exists():
        download_pdfs_from_list(config=cfg)

    builder = DatasetBuilder(tokenizer_name=args.model_name, max_seq_length=2048, config=cfg)
    records = builder.process_dataset_folder()

    if records > 0:
        builder.pack_sequences_for_dapt()
    else:
        logger.info(f"No PDFs/text found in {cfg.dataset_dir}. Place your books/articles or down.txt in the folder.")
