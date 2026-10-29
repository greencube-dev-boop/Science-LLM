"""
Stage 3: Evaluation, Model Merging & Versioned Export
=====================================================
Merges LoRA adapter into base model and exports standalone weights to:
`drive/Science LLM/models/SCLLM-v<version>`

Synchronizes benchmark evaluation and model records into:
`drive/Science LLM/config/data.json`
"""

import os
import torch
import logging
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config_manager import ScienceLLMConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ExportAndEvaluate")

SCIENCE_BENCHMARK_PROMPTS = [
    {
        "test_name": "crispr_cas9_mechanism",
        "domain": "Biochemistry",
        "prompt": "Explain the biochemical mechanism of CRISPR-Cas9 in targeted DNA cleavage, specifically how the PAM sequence and guide RNA facilitate Cas9 activation."
    },
    {
        "test_name": "quantum_matter_waves",
        "domain": "Quantum Physics",
        "prompt": "Derive the relation between phase velocity and group velocity for a de Broglie matter wave, and explain why group velocity matches the classical particle velocity."
    },
    {
        "test_name": "diels_alder_stereochemistry",
        "domain": "Organic Chemistry",
        "prompt": "Describe the mechanism of the Diels-Alder reaction between 1,3-butadiene and maleic anhydride, noting the stereochemical outcome (endo rule) and frontier molecular orbital interactions."
    }
]


def evaluate_and_log(cfg: ScienceLLMConfig, model, tokenizer, device: str = "cuda"):
    """Runs scientific evaluation and logs results to data.json."""
    logger.info("Running Qualitative Scientific Domain Evaluation...")
    model.eval()

    for idx, item in enumerate(SCIENCE_BENCHMARK_PROMPTS, 1):
        domain = item["domain"]
        query = item["prompt"]
        test_name = item["test_name"]

        formatted_input = (
            f"<|im_start|>system\nYou are an expert scientific researcher and AI assistant.<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = tokenizer(formatted_input, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\n{'='*70}\n[TEST {idx} - {domain}]\nPrompt: {query}\n\nGenerated Response:\n{response.strip()}\n{'='*70}\n")

        # Save to data.json
        cfg.log_evaluation_result(
            test_name=test_name,
            score=None,
            details={
                "domain": domain,
                "prompt": query,
                "response": response.strip()
            }
        )


def merge_and_export(
    cfg: ScienceLLMConfig,
    base_model_name: str,
    adapter_path: Optional[str] = None,
    version: Optional[str] = None
):
    """
    Merges LoRA adapter into 16-bit base model and saves to:
    `drive/Science LLM/models/SCLLM-v<version>`
    """
    if version:
        cfg.update_version(version)

    current_ver = cfg.state.get("current_version", "1.0.0")
    save_dir = cfg.get_model_save_dir(version=current_ver)

    if not adapter_path:
        # Check Stage 2 SFT adapter first, then Stage 1 DAPT
        sft_adapter = cfg.checkpoints_dir / "stage2_sft" / "final_sft_adapter"
        dapt_adapter = cfg.checkpoints_dir / "stage1_dapt" / "final_dapt_adapter"
        adapter_path = str(sft_adapter if sft_adapter.exists() else dapt_adapter)

    logger.info(f"Target model save destination: {save_dir}")
    logger.info(f"Loading unquantized FP16 base model: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )

    logger.info(f"Applying LoRA adapter: {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    logger.info("Merging LoRA weights into base model layers...")
    merged_model = model.merge_and_unload()

    logger.info(f"Saving merged 16-bit standalone model to: {save_dir}...")
    merged_model.save_pretrained(str(save_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(save_dir))

    cfg.state["trained_models"] = cfg.state.get("trained_models", [])
    cfg.state["trained_models"].append({
        "tag": f"SCLLM-v{current_ver}",
        "version": current_ver,
        "path": str(save_dir),
        "base_model": base_model_name,
        "adapter_source": adapter_path
    })
    cfg.save_state()
    logger.info(f"Successfully saved and registered SCLLM-v{current_ver}!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export & Evaluate Science LLM")
    parser.add_argument("--base_dir", type=str, default=None, help="Base Drive directory")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--version", type=str, default=None, help="Version number (e.g. 1.0.0)")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation without merging")
    args = parser.parse_args()

    config = ScienceLLMConfig(base_dir=args.base_dir)

    if not args.eval_only:
        merge_and_export(config, base_model_name=args.base_model, version=args.version)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_dir = config.get_model_save_dir(version=args.version)
    if target_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(target_dir))
        model = AutoModelForCausalLM.from_pretrained(str(target_dir), torch_dtype=torch.float16, device_map="auto")
        evaluate_and_log(config, model, tokenizer, device=device)
