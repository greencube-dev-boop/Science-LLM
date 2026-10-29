# Science-Focused Domain LLM (SCLLM)

A specialized training, evaluation, and export pipeline to adapt open-source LLMs into science research models on a Google Colab Tesla T4 GPU (16GB VRAM).

---

## 📂 Google Drive Directory Structure

All persistent training data, checkpoints, models, and metadata are automatically organized under your Google Drive root:

```
drive/MyDrive/Science LLM/
├── down.txt                 # (Optional) Text file with custom PDF download links
│
├── Dataset/                 # Downloaded PDFs and textbooks
│   ├── extracted_texts/     # Pre-converted clean .txt files with formulas preserved
│   ├── processed_science_corpus.jsonl
│   └── tokenized_science_blocks/
│
├── checkpoints/             # Saved training steps & adapters
│   ├── stage1_dapt/         # Stage 1 Continued Pre-Training checkpoints
│   │   └── final_dapt_adapter/
│   └── stage2_sft/          # Stage 2 Supervised Fine-Tuning checkpoints
│       └── final_sft_adapter/
│
├── models/                  # Exported standalone merged models
│   ├── SCLLM-v1.0.0/        # Merged 16-bit standalone weights
│   └── SCLLM-v1.1.0/
│
└── config/                  # State tracking, restart/resume data, and evaluation logs
    ├── data.json            # Central experiment & run metadata tracker
    └── hash_db.json         # SHA-256 database for zero-duplicate storage
```

---

## 🔬 Science Paper & Book Harvester (`science_harvester.py`)

Harvests 1,000 to 5,000+ open-access research papers and books directly from **Europe PMC** and **arXiv** using domain tags:
- **Default Tags**: `organic chemistry`, `biochemistry`, `botany`, `plant biology`, `chemistry`, `biology`
- **SHA-256 Deduplication**: Every file's content hash is recorded in `drive/MyDrive/Science LLM/config/hash_db.json`. Any duplicate paper or mirror is automatically identified and removed.
- **Header Verification**: Ensures every file contains valid `%PDF` binary magic bytes.

CLI execution:
```bash
python science_harvester.py --limit 1000 --tags "organic chemistry" "biochemistry" "botany" "plant biology" "chemistry" "biology"
```

---

## 📄 PDF to Clean Text Batch Converter (`pdf_converter.py`)

Converting PDFs into text ahead of time provides 50x–100x faster tokenization and eliminates runtime memory spikes:
- Multi-column layout reconstruction via PyMuPDF.
- Preserves inline and display mathematical formulas (`$...$`, `$$...$$`).
- Truncates bibliographies/reference lists and removes publisher header noise.
- Outputs clean text to `drive/MyDrive/Science LLM/Dataset/extracted_texts/`.

CLI execution:
```bash
# Convert all PDFs to TXT (keep PDFs)
python pdf_converter.py --workers 4

# Or convert and delete raw PDFs to save Google Drive storage:
python pdf_converter.py --workers 4 --delete_pdfs
```

---

## 📋 State Tracking: `drive/Science LLM/config/data.json`

The [`config_manager.py`](file:///d:/code/LLM/config_manager.py) module manages `data.json` with atomic writes, recording:
- **Active Model Version**: `SCLLM-v<version>` (e.g., `SCLLM-v1.0.0`).
- **Harvesting Logs**: Number of papers fetched per domain tag.
- **Conversion Stats**: Total PDFs scanned and converted into clean text.
- **Checkpoints & Resumption**: Latest checkpoint step, stage, directory path, and `can_resume` flag to automatically restart from the exact step if Colab times out.
- **Training Runs**: Start/end timestamps, duration, loss, hyperparameters (batch size, learning rate, gradient accumulation).
- **Evaluation Benchmarks**: Test prompts, domain tags, generated responses, and benchmark evaluation history.

---

## 🛠️ Repository Files

- [`science_harvester.py`](file:///d:/code/LLM/science_harvester.py): Open scientific paper downloader querying Europe PMC & arXiv by tags with multithreading.
- [`hash_db.py`](file:///d:/code/LLM/hash_db.py): SHA-256 content deduplication manager.
- [`pdf_converter.py`](file:///d:/code/LLM/pdf_converter.py): Fast batch PDF-to-TXT converter with formula preservation.
- [`downloader.py`](file:///d:/code/LLM/downloader.py): Stream downloader for custom links in `down.txt`.
- [`config_manager.py`](file:///d:/code/LLM/config_manager.py): Central configuration manager for paths, versions, and atomic `data.json` synchronization.
- [`data_pipeline.py`](file:///d:/code/LLM/data_pipeline.py): Token chunk packer prioritizing pre-converted text files.
- [`train_continued_pretrain.py`](file:///d:/code/LLM/train_continued_pretrain.py): Stage 1 Domain-Adaptive Continued Pre-Training with Unsloth / QLoRA 4-bit and auto-resume callback.
- [`train_sft.py`](file:///d:/code/LLM/train_sft.py): Stage 2 Supervised Fine-Tuning with ChatML prompt formatting and assistant completion masking.
- [`export_and_evaluate.py`](file:///d:/code/LLM/export_and_evaluate.py): Merges LoRA adapters into standalone 16-bit weights at `models/SCLLM-v<version>` and logs evaluation tests.
- [`Science_LLM_Colab_Master.ipynb`](file:///d:/code/LLM/Science_LLM_Colab_Master.ipynb): Complete end-to-end Google Colab master notebook.
- [`requirements.txt`](file:///d:/code/LLM/requirements.txt): Pinned dependencies for T4 GPU environments.
- [`.gitignore`](file:///d:/code/LLM/.gitignore): Excludes large model binaries, virtualenvs, and temporary files from version control.

---

## 🚀 Quickstart in Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Open [`Science_LLM_Colab_Master.ipynb`](file:///d:/code/LLM/Science_LLM_Colab_Master.ipynb) or load directly from GitHub: `https://colab.research.google.com/github/greencube-dev-boop/Science-LLM/blob/main/Science_LLM_Colab_Master.ipynb`.
3. Set runtime to **GPU (T4)**.
4. Run cells sequentially:
   - Step 5 will harvest open papers across the science tags.
   - Step 7 converts all PDFs to clean text.
   - Steps 9-12 train, evaluate, and export the model to `SCLLM-v1.0.0`.
