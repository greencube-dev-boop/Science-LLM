"""
Stage 3: Evaluation, Model Merging & Versioned Export
=====================================================
Features:
1. Quantitative multiple-choice benchmark evaluation on SciQ science exam validation set (calculates Accuracy %).
2. In-depth qualitative evaluation across all 6 core science tags:
   - Organic Chemistry, Biochemistry, Botany, Plant Biology, General Chemistry, Cellular Biology.
3. LoRA adapter merge into standalone 16-bit model at:
   `drive/Science LLM/models/SCLLM-v<version>`
4. GGUF quantization instructions for Ollama / LM Studio.
5. All scores and outputs synchronized to `drive/Science LLM/config/data.json`.
"""

import os
import re
import time
import torch
import logging
from typing import Optional, List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from config_manager import ScienceLLMConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ExportAndEvaluate")

SCIENCE_TAG_BENCHMARK_PROMPTS = [
    {
        "test_name": "organic_chem_diels_alder",
        "domain": "Organic Chemistry",
        "prompt": "Explain the stereochemical outcome of the Diels-Alder cycloaddition between cyclopentadiene and maleic anhydride. Detail the secondary orbital interactions responsible for the endo rule."
    },
    {
        "test_name": "biochem_crispr_pam",
        "domain": "Biochemistry",
        "prompt": "Detail the molecular mechanism of Cas9 endonuclease activation. How do the PAM (Protospacer Adjacent Motif) sequence and sgRNA seed region cooperatively trigger DNA target unwinding and dual-strand cleavage?"
    },
    {
        "test_name": "botany_c4_cam_photosynthesis",
        "domain": "Botany & Plant Biology",
        "prompt": "Compare Kranz anatomy in C4 plants with the temporal carboxylation separation in CAM plants. Explain how both adaptations mitigate RuBisCO oxygenase activity and photorespiration in arid climates."
    },
    {
        "test_name": "chemistry_crystal_field_theory",
        "domain": "Chemistry",
        "prompt": "Derive the d-orbital splitting pattern for an octahedral transition metal complex under Crystal Field Theory. Explain high-spin versus low-spin electron configurations in d5 complexes."
    },
    {
        "test_name": "biology_atp_synthase_motor",
        "domain": "Cellular Biology",
        "prompt": "Describe the rotary motor mechanism of the F0F1 ATP synthase complex. How does proton translocating through the c-ring drive conformational changes in the alpha3beta3 catalytic head to synthesize ATP?"
    }
]


def run_quantitative_sciq_benchmark(model, tokenizer, num_samples: int = 50, device: str = "cuda") -> float:
    """Evaluates multiple-choice accuracy on the SciQ science validation set."""
    logger.info(f"Running Quantitative Scientific Benchmark on SciQ (N={num_samples})...")
    sciq_val = load_dataset("sciq", split=f"validation[:{num_samples}]")

    correct = 0
    total = len(sciq_val)

    for item in sciq_val:
        question = item["question"]
        correct_answer = item["correct_answer"].strip()
        support = item.get("support", "").strip()

        context_str = f"Context: {support}\n" if support else ""
        formatted_prompt = (
            f"<|im_start|>system\nYou are a scientific assistant. Answer concisely with only the exact scientific term or fact.<|im_end|>\n"
            f"<|im_start|>user\n{context_str}{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                temperature=0.1,
                eos_token_id=tokenizer.eos_token_id
            )

        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        if correct_answer.lower() in gen_text.lower():
            correct += 1

    accuracy = (correct / max(1, total)) * 100.0
    logger.info(f"SciQ Benchmark Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    return accuracy


def evaluate_qualitative_prompts(cfg: ScienceLLMConfig, model, tokenizer, device: str = "cuda"):
    """Runs scientific domain prompts and logs structured responses into data.json."""
    logger.info("Running Qualitative Scientific Domain Prompts across Core Tags...")
    model.eval()

    for idx, item in enumerate(SCIENCE_TAG_BENCHMARK_PROMPTS, 1):
        domain = item["domain"]
        query = item["prompt"]
        test_name = item["test_name"]

        formatted_input = (
            f"<|im_start|>system\nYou are an expert scientific researcher and AI assistant. Provide rigorous, step-by-step reasoning using standard LaTeX and precise nomenclature.<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = tokenizer(formatted_input, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=350,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"\n{'='*75}\n[TEST {idx} - {domain}]\nPrompt: {query}\n\nGenerated Response:\n{response}\n{'='*75}\n")

        cfg.log_evaluation_result(
            test_name=test_name,
            score=None,
            details={
                "domain": domain,
                "prompt": query,
                "response": response
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
        # Automatically find the latest adapter (SFT preferred, otherwise DAPT)
        sft_adapter = cfg.get_stage_adapter("stage2_sft")
        dapt_adapter = cfg.get_stage_adapter("stage1_dapt")
        adapter_path = str(sft_adapter if sft_adapter else dapt_adapter)

    logger.info(f"Target model destination: {save_dir}")
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
        "adapter_source": adapter_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    cfg.save_state()
    logger.info(f"Successfully saved and registered SCLLM-v{current_ver}!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export & Evaluate Science LLM")
    parser.add_argument("--base_dir", type=str, default=None, help="Base Drive directory")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--version", type=str, default=None, help="Version tag (e.g. 1.0.0)")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation without merging")
    parser.add_argument("--sciq_samples", type=int, default=50, help="Number of SciQ validation questions")
    args = parser.parse_args()

    config = ScienceLLMConfig(base_dir=args.base_dir)

    if not args.eval_only:
        merge_and_export(config, base_model_name=args.base_model, version=args.version)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_dir = config.get_model_save_dir(version=args.version)
    if target_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(target_dir))
        model = AutoModelForCausalLM.from_pretrained(str(target_dir), torch_dtype=torch.float16, device_map="auto")
        
        # 1. Multiple-choice Quantitative Benchmark
        accuracy = run_quantitative_sciq_benchmark(model, tokenizer, num_samples=args.sciq_samples, device=device)
        config.log_evaluation_result("sciq_benchmark_accuracy", score=round(accuracy, 2), details={"num_samples": args.sciq_samples})

        # 2. Qualitative Domain Test Battery
        evaluate_qualitative_prompts(config, model, tokenizer, device=device)
