"""
Stage 1: Domain-Adaptive Continued Pre-Training (DAPT)
======================================================
Optimized for Google Colab T4 GPU (16GB VRAM) with automatic resume support.

Paths:
- Dataset: drive/Science LLM/Dataset
- Checkpoints: drive/Science LLM/checkpoints/stage1_dapt
- Metadata State: drive/Science LLM/config/data.json
"""

import os
import sys
import time
import logging

# Ensure unsloth is imported before transformers/peft to apply memory optimizations cleanly
HAS_UNSLOTH = False
try:
    import unsloth
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except Exception:
    HAS_UNSLOTH = False

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk, load_dataset
from config_manager import ScienceLLMConfig, create_training_args

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainDAPT")


class CheckpointSyncCallback(TrainerCallback):
    """Syncs checkpoint state and loss metrics into data.json at each save step."""

    def __init__(self, config: ScienceLLMConfig, stage: str = "stage1_dapt"):
        self.config = config
        self.stage = stage

    def on_save(self, args, state, control, **kwargs):
        latest_checkpoint = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        last_log = state.log_history[-1] if state.log_history else {}
        loss = last_log.get("loss", None)

        self.config.register_checkpoint(
            stage=self.stage,
            step=state.global_step,
            checkpoint_path=latest_checkpoint,
            loss=loss,
            metrics={"epoch": state.epoch, "learning_rate": last_log.get("learning_rate")}
        )


def is_unsloth_available() -> bool:
    global HAS_UNSLOTH
    return HAS_UNSLOTH


def run_training_unsloth(cfg: ScienceLLMConfig, model_name: str, max_seq_length: int = 2048, epochs: int = 1, batch_size: int = 2, grad_accum: int = 8, lr: float = 1e-4):
    from trl import SFTTrainer

    output_dir = str(cfg.checkpoints_dir / "stage1_dapt")
    resume_checkpoint = cfg.get_resume_checkpoint(stage="stage1_dapt")

    logger.info(f"Loading {model_name} via Unsloth with 4-bit NF4...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=torch.float16,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset_path = cfg.dataset_dir / "tokenized_science_blocks"
    if dataset_path.exists():
        logger.info(f"Loading pre-tokenized dataset from: {dataset_path}")
        train_dataset = load_from_disk(str(dataset_path))
    else:
        logger.info(f"No local blocks found. Falling back to SciQ science dataset...")
        train_dataset = load_dataset("sciq", split="train[:5000]")

    training_args = create_training_args(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=0.05,
        num_train_epochs=epochs,
        learning_rate=lr,
        fp16=True,
        bf16=False,
        logging_steps=10,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        seed=3407,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text" if "text" in train_dataset.column_names else None,
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=True if "text" in train_dataset.column_names else False,
        args=training_args,
        callbacks=[CheckpointSyncCallback(cfg, stage="stage1_dapt")]
    )

    start_time = time.time()
    logger.info(f"Starting Stage 1 training (Resume checkpoint: {resume_checkpoint})...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    adapter_save_dir = os.path.join(output_dir, "final_dapt_adapter")
    model.save_pretrained(adapter_save_dir)
    tokenizer.save_pretrained(adapter_save_dir)

    cfg.log_training_run({
        "stage": "stage1_dapt",
        "duration_seconds": round(time.time() - start_time, 2),
        "adapter_path": adapter_save_dir,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "status": "completed"
    })
    logger.info(f"Stage 1 complete! Saved adapter to: {adapter_save_dir}")


def run_training_standard_peft(cfg: ScienceLLMConfig, model_name: str, epochs: int = 1, batch_size: int = 1, grad_accum: int = 16, lr: float = 1e-4):
    output_dir = str(cfg.checkpoints_dir / "stage1_dapt")
    resume_checkpoint = cfg.get_resume_checkpoint(stage="stage1_dapt")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading base model {model_name} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    dataset_path = cfg.dataset_dir / "tokenized_science_blocks"
    if dataset_path.exists():
        train_dataset = load_from_disk(str(dataset_path))
    else:
        logger.info("Using SciQ dataset as fallback...")
        raw = load_dataset("sciq", split="train[:5000]")
        train_dataset = raw.map(lambda x: {"text": f"Topic: {x['question']}\nDetails: {x['support']}\nAnswer: {x['correct_answer']}"})

    training_args = create_training_args(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=0.05,
        num_train_epochs=epochs,
        learning_rate=lr,
        fp16=True,
        bf16=False,
        logging_steps=10,
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[CheckpointSyncCallback(cfg, stage="stage1_dapt")]
    )

    start_time = time.time()
    logger.info(f"Starting Stage 1 training (Resume checkpoint: {resume_checkpoint})...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    adapter_save_dir = os.path.join(output_dir, "final_dapt_adapter")
    model.save_pretrained(adapter_save_dir)
    tokenizer.save_pretrained(adapter_save_dir)

    cfg.log_training_run({
        "stage": "stage1_dapt",
        "duration_seconds": round(time.time() - start_time, 2),
        "adapter_path": adapter_save_dir,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "status": "completed"
    })
    logger.info(f"Stage 1 complete! Saved adapter to: {adapter_save_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1 DAPT Training")
    parser.add_argument("--base_dir", type=str, default=None, help="Base Drive directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    config = ScienceLLMConfig(base_dir=args.base_dir)

    if is_unsloth_available():
        run_training_unsloth(config, model_name=args.model_name, epochs=args.epochs, lr=args.lr)
    else:
        run_training_standard_peft(config, model_name=args.model_name, epochs=args.epochs, lr=args.lr)
