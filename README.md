# Science-Focused Domain LLM (SCLLM)

A specialized training, evaluation, and export pipeline to adapt open-source LLMs into science research models on a Google Colab Tesla T4 GPU (16GB VRAM).

---

## 📂 Google Drive Directory Structure

All persistent training data, checkpoints, models, and metadata are automatically organized under your Google Drive root:

```
drive/MyDrive/Science LLM/
├── down.txt                 # (Optional) Text file with one PDF download link per line
│
├── Dataset/                 # Input PDFs, research papers, textbooks, and text files (.pdf, .txt, .md)
│   ├── processed_science_corpus.jsonl
│   └── tokenized_science_blocks/
│
├── checkpoints/             # Saved training steps & adapters
│   ├── stage1_dapt/         # Stage 1 Continued Pre-Training checkpoints (e.g. checkpoint-100, checkpoint-200)
│   │   └── final_dapt_adapter/
│   └── stage2_sft/          # Stage 2 Supervised Fine-Tuning checkpoints
│       └── final_sft_adapter/
│
├── models/                  # Exported standalone merged models
│   ├── SCLLM-v1.0.0/        # Merged 16-bit standalone weights
│   └── SCLLM-v1.1.0/
│
└── config/                  # State tracking, restart/resume data, and evaluation logs
    └── data.json            # Central metadata file
```

---

## 📥 Automated PDF Downloading (`down.txt`)

You can create a `down.txt` file directly inside `drive/MyDrive/Science LLM/down.txt` containing one download URL per line:
```text
https://arxiv.org/pdf/1706.03762.pdf
https://openstax.org/downloads/college-physics.pdf
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/pdf/main.pdf
```

The pipeline automatically:
1. Streams each file into `drive/MyDrive/Science LLM/Dataset/`.
2. Validates the `%PDF` magic byte header (preventing paywall HTML pages from being saved as corrupted PDFs).
3. Skips already downloaded files to conserve bandwidth.
4. Records download metrics into `drive/MyDrive/Science LLM/config/data.json`.

You can also run the downloader standalone:
```bash
python downloader.py --down_file "/content/drive/MyDrive/Science LLM/down.txt"
```

---

## 📋 State Tracking: `drive/Science LLM/config/data.json`

The [`config_manager.py`](file:///d:/code/LLM/config_manager.py) module manages `data.json` with atomic writes, recording:
- **Active Model Version**: `SCLLM-v<version>` (e.g., `SCLLM-v1.0.0`).
- **Download Records**: Total URLs found and successfully downloaded from `down.txt`.
- **Checkpoints & Resumption**: Latest checkpoint step, stage, directory path, and `can_resume` flag to automatically restart from the exact step if Colab times out.
- **Training Runs**: Start/end timestamps, duration, loss, hyperparameters (batch size, learning rate, gradient accumulation).
- **Evaluation Benchmarks**: Test prompts, domain tags, generated responses, and benchmark evaluation history.

---

## 🛠️ Repository Files

- [`downloader.py`](file:///d:/code/LLM/downloader.py): Stream downloader with URL parsing, `%PDF` verification, and duplicate skipping.
- [`config_manager.py`](file:///d:/code/LLM/config_manager.py): Central configuration manager for paths, versions, and atomic `data.json` synchronization.
- [`data_pipeline.py`](file:///d:/code/LLM/data_pipeline.py): Multi-column PDF parsing (PyMuPDF `fitz`), LaTeX formula preservation, reference trimming, and token chunk packing.
- [`train_continued_pretrain.py`](file:///d:/code/LLM/train_continued_pretrain.py): Stage 1 Domain-Adaptive Continued Pre-Training with Unsloth / QLoRA 4-bit and auto-resume callback.
- [`train_sft.py`](file:///d:/code/LLM/train_sft.py): Stage 2 Supervised Fine-Tuning with ChatML prompt formatting and assistant completion masking.
- [`export_and_evaluate.py`](file:///d:/code/LLM/export_and_evaluate.py): Merges LoRA adapters into standalone 16-bit weights at `models/SCLLM-v<version>` and logs evaluation tests.
- [`Science_LLM_Colab_Master.ipynb`](file:///d:/code/LLM/Science_LLM_Colab_Master.ipynb): Complete end-to-end Google Colab notebook with automatic Drive mounting, resume logic, and testing.
- [`requirements.txt`](file:///d:/code/LLM/requirements.txt): Pinned dependencies for T4 GPU environments.
- [`.gitignore`](file:///d:/code/LLM/.gitignore): Excludes large model binaries, virtualenvs, and temporary files from version control.

---

## 🚀 Quickstart in Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Upload [`Science_LLM_Colab_Master.ipynb`](file:///d:/code/LLM/Science_LLM_Colab_Master.ipynb) or open directly from GitHub: `https://colab.research.google.com/github/greencube-dev-boop/Science-LLM/blob/main/Science_LLM_Colab_Master.ipynb`.
3. Set runtime to **GPU (T4)**.
4. Run all cells sequentially.
