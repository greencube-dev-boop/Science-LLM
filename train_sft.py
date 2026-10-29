"""
Stage 2: Supervised Fine-Tuning (SFT) & Scientific Reasoning Alignment
======================================================================
Fine-tunes the Stage 1 DAPT model into an interactive scientific reasoning assistant.

Paths:
- Input DAPT Adapter: auto-discovered via cfg.get_stage_adapter("stage1_dapt")
- Output SFT Checkpoints: drive/Science LLM/checkpoints/stage2_sft
- State Metadata: drive/Science LLM/config/data.json
"""

import os
import time
import logging
from typing import Optional

# Ensure unsloth is imported first if present
try:
    import unsloth
except Exception:
    pass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from config_manager import ScienceLLMConfig, create_training_args

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainSFT")

SYSTEM_PROMPT = """You are an expert scientific researcher and AI assistant specialized in STEM fields including physics, chemistry, biology, medicine, and mathematics.
When answering scientific questions:
1. Provide accurate, evidence-based reasoning step-by-step.
2. Use precise nomenclature and standard LaTeX notation for mathematical equations and chemical formulas where appropriate.
3. Distinguish clearly between established scientific consensus and theoretical hypotheses."""


class SFTCheckpointSyncCallback(TrainerCallback):
    """Syncs checkpoint state into data.json at each save step."""

    def __init__(self, config: ScienceLLMConfig):
        self.config = config

    def on_save(self, args, state, control, **kwargs):
        latest_checkpoint = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        last_log = state.log_history[-1] if state.log_history else {}
        loss = last_log.get("loss", None)

        self.config.register_checkpoint(
            stage="stage2_sft",
            step=state.global_step,
            checkpoint_path=latest_checkpoint,
            loss=loss,
            metrics={"epoch": state.epoch, "learning_rate": last_log.get("learning_rate")}
        )


def format_chatml_prompt(example) -> str:
    question = example.get("question", "")
    context = example.get("support", "")
    answer = example.get("correct_answer", "")

    user_query = f"{question}\n\nContext:\n{context}" if context else question
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>"
    )


def train_sft(cfg: ScienceLLMConfig, base_model_name: str = "Qwen/Qwen2.5-3B", epochs: int = 2, batch_size: int = 2, grad_accum: int = 8, lr: float = 2e-4):
    output_dir = str(cfg.checkpoints_dir / "stage2_sft")
    resume_checkpoint = cfg.get_resume_checkpoint(stage="stage2_sft")
    
    # Automatically locate Stage 1 adapter (final_dapt_adapter or latest checkpoint-*)
    dapt_adapter_path = cfg.get_stage_adapter(stage="stage1_dapt")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading base model {base_model_name} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    if dapt_adapter_path and os.path.exists(dapt_adapter_path):
        logger.info(f"Loading Stage 1 DAPT adapter from: {dapt_adapter_path}")
        model = PeftModel.from_pretrained(model, dapt_adapter_path, is_trainable=True)
    else:
        logger.info("No Stage 1 adapter found on disk. Initializing fresh LoRA adapters for SFT...")
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

    # Load Science Instruction Dataset
    sciq_dataset = load_dataset("sciq", split="train")
    formatted_dataset = sciq_dataset.map(lambda ex: {"text": format_chatml_prompt(ex)})

    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

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
        lr_scheduler_type="cosine",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        data_collator=collator,
        max_seq_length=2048,
        args=training_args,
        callbacks=[SFTCheckpointSyncCallback(cfg)]
    )

    start_time = time.time()
    logger.info(f"Starting Stage 2 SFT training (Resume checkpoint: {resume_checkpoint})...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    final_sft_adapter = os.path.join(output_dir, "final_sft_adapter")
    model.save_pretrained(final_sft_adapter)
    tokenizer.save_pretrained(final_sft_adapter)

    cfg.log_training_run({
        "stage": "stage2_sft",
        "duration_seconds": round(time.time() - start_time, 2),
        "adapter_path": final_sft_adapter,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "status": "completed"
    })
    logger.info(f"Stage 2 complete! Saved adapter to: {final_sft_adapter}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2 SFT Training")
    parser.add_argument("--base_dir", type=str, default=None, help="Base Drive directory")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    config = ScienceLLMConfig(base_dir=args.base_dir)
    train_sft(config, base_model_name=args.base_model, epochs=args.epochs, lr=args.lr)
