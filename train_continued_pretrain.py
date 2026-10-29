"""
Stage 1: Domain-Adaptive Continued Pre-Training (DAPT)
======================================================
Optimized for Google Colab T4 GPU (16GB VRAM) with automatic resume support.

Enhancements:
- Scaled Dot-Product Attention (SDPA) hardware acceleration.
- Higher expressive LoRA adapter rank (r=32, alpha=64) for dense domain absorption.
- Perplexity (PPL = exp(loss)) tracking in data.json.
- Paged AdamW 8-bit optimizer and gradient checkpointing.
"""

import os
import sys
import time
import math
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
    """Syncs checkpoint state, loss, and perplexity metrics into data.json at each save step."""

    def __init__(self, config: ScienceLLMConfig, stage: str = "stage1_dapt"):
        self.config = config
        self.stage = stage

    def on_save(self, args, state, control, **kwargs):
        latest_checkpoint = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        last_log = state.log_history[-1] if state.log_history else {}
        loss = last_log.get("loss", None)
        ppl = math.exp(loss) if (loss is not None and loss < 100) else None

        self.config.register_checkpoint(
            stage=self.stage,
            step=state.global_step,
            checkpoint_path=latest_checkpoint,
            loss=loss,
            metrics={
                "epoch": state.epoch,
                "learning_rate": last_log.get("learning_rate"),
                "perplexity": round(ppl, 4) if ppl else None
            }
        )


def is_unsloth_available() -> bool:
    global HAS_UNSLOTH
    return HAS_UNSLOTH


def run_training_unsloth(
    cfg: ScienceLLMConfig,
    model_name: str,
    max_seq_length: int = 2048,
    epochs: int = 1,
    batch_size: int = 2,
    grad_accum: int = 8,
    lr: float = 1e-4,
    lora_r: int = 32,
    lora_alpha: int = 64
):
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
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
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
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    adapter_save_dir = os.path.join(output_dir, "final_dapt_adapter")
    model.save_pretrained(adapter_save_dir)
    tokenizer.save_pretrained(adapter_save_dir)

    train_loss = train_result.training_loss if hasattr(train_result, "training_loss") else None
    final_ppl = math.exp(train_loss) if (train_loss is not None and train_loss < 100) else None

    cfg.log_training_run({
        "stage": "stage1_dapt",
        "duration_seconds": round(time.time() - start_time, 2),
        "adapter_path": adapter_save_dir,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "final_loss": train_loss,
        "final_perplexity": round(final_ppl, 4) if final_ppl else None,
        "status": "completed"
    })
    logger.info(f"Stage 1 complete! Saved adapter to: {adapter_save_dir} (Final Loss: {train_loss:.4f}, Perplexity: {final_ppl:.2f})")


def run_training_standard_peft(
    cfg: ScienceLLMConfig,
    model_name: str,
    epochs: int = 1,
    batch_size: int = 1,
    grad_accum: int = 16,
    lr: float = 1e-4,
    lora_r: int = 32,
    lora_alpha: int = 64
):
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

    logger.info(f"Loading base model {model_name} in 4-bit with SDPA acceleration...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
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
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    adapter_save_dir = os.path.join(output_dir, "final_dapt_adapter")
    model.save_pretrained(adapter_save_dir)
    tokenizer.save_pretrained(adapter_save_dir)

    train_loss = train_result.training_loss if hasattr(train_result, "training_loss") else None
    final_ppl = math.exp(train_loss) if (train_loss is not None and train_loss < 100) else None

    cfg.log_training_run({
        "stage": "stage1_dapt",
        "duration_seconds": round(time.time() - start_time, 2),
        "adapter_path": adapter_save_dir,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "final_loss": train_loss,
        "final_perplexity": round(final_ppl, 4) if final_ppl else None,
        "status": "completed"
    })
    logger.info(f"Stage 1 complete! Saved adapter to: {adapter_save_dir} (Loss: {train_loss:.4f}, PPL: {final_ppl:.2f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1 DAPT Training")
    parser.add_argument("--base_dir", type=str, default=None, help="Base Drive directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    args = parser.parse_args()

    config = ScienceLLMConfig(base_dir=args.base_dir)

    if is_unsloth_available():
        run_training_unsloth(config, model_name=args.model_name, epochs=args.epochs, lr=args.lr, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    else:
        run_training_standard_peft(config, model_name=args.model_name, epochs=args.epochs, lr=args.lr, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
