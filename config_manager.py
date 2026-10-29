"""
Config & State Manager for Science LLM
=======================================
Manages experiment metadata, dataset tracking, checkpoint state,
and versioning in `drive/Science LLM/config/data.json`.

Ensures seamless resumption and recovery across Google Colab session timeouts.
"""

import os
import json
import time
import shutil
import inspect
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ConfigManager")


def create_training_args(**kwargs):
    """
    Creates a transformers.TrainingArguments instance compatible across all transformers versions.
    - Handles warmup_ratio -> warmup_steps migration in transformers v5+.
    - Handles evaluation_strategy -> eval_strategy migration.
    - Filters unsupported kwargs to prevent TypeErrors across varying library versions.
    """
    from transformers import TrainingArguments

    sig = inspect.signature(TrainingArguments.__init__)
    valid_params = set(sig.parameters.keys())

    # Handle warmup_ratio deprecation in transformers v5+
    if "warmup_ratio" in kwargs and "warmup_ratio" not in valid_params:
        ratio = kwargs.pop("warmup_ratio")
        if "warmup_steps" in valid_params and "warmup_steps" not in kwargs:
            kwargs["warmup_steps"] = ratio

    # Handle evaluation_strategy -> eval_strategy migration
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in valid_params:
        strat = kwargs.pop("evaluation_strategy")
        if "eval_strategy" in valid_params:
            kwargs["eval_strategy"] = strat

    # Filter to only valid kwargs
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}
    return TrainingArguments(**filtered)


class ScienceLLMConfig:
    """Manages paths and persistent state in data.json for Science LLM."""

    DEFAULT_COLAB_BASE = "/content/drive/MyDrive/Science LLM"
    DEFAULT_LOCAL_BASE = "./drive/Science LLM"

    def __init__(self, base_dir: Optional[str] = None, version: str = "1.0.0"):
        if base_dir:
            self.base_dir = Path(base_dir)
        elif os.path.exists("/content/drive/MyDrive"):
            self.base_dir = Path(self.DEFAULT_COLAB_BASE)
        else:
            self.base_dir = Path(self.DEFAULT_LOCAL_BASE)

        self.dataset_dir = self.base_dir / "Dataset"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.models_dir = self.base_dir / "models"
        self.config_dir = self.base_dir / "config"
        self.data_json_path = self.config_dir / "data.json"

        self.ensure_directories()
        self.state = self.load_or_init_state(default_version=version)

    def ensure_directories(self):
        """Creates the required folder hierarchy if it does not exist."""
        for d in [self.dataset_dir, self.checkpoints_dir, self.models_dir, self.config_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_model_save_dir(self, version: Optional[str] = None) -> Path:
        """Returns target directory for trained model: SCLLM-v<version>."""
        ver = version or self.state.get("current_version", "1.0.0")
        tag = f"SCLLM-v{ver}"
        target = self.models_dir / tag
        target.mkdir(parents=True, exist_ok=True)
        return target

    def load_or_init_state(self, default_version: str = "1.0.0") -> Dict[str, Any]:
        """Loads data.json or initializes it with schema."""
        if self.data_json_path.exists():
            try:
                with open(self.data_json_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    logger.info(f"Loaded existing state from {self.data_json_path} (Version: {state.get('current_version')})")
                    return state
            except Exception as e:
                logger.warning(f"Error reading data.json: {e}. Backing up and initializing new state.")
                backup_path = self.config_dir / f"data.json.bak.{int(time.time())}"
                shutil.copyfile(self.data_json_path, backup_path)

        initial_state = {
            "project_name": "Science LLM",
            "current_version": default_version,
            "model_name_tag": f"SCLLM-v{default_version}",
            "base_model": "Qwen/Qwen2.5-3B",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "paths": {
                "base_dir": str(self.base_dir),
                "dataset_dir": str(self.dataset_dir),
                "checkpoints_dir": str(self.checkpoints_dir),
                "models_dir": str(self.models_dir),
                "config_dir": str(self.config_dir)
            },
            "checkpoints": {
                "latest_stage": None,
                "latest_step": 0,
                "latest_checkpoint_path": None,
                "can_resume": False,
                "history": []
            },
            "training_runs": [],
            "evaluation_benchmarks": {
                "latest_eval_time": None,
                "test_results": []
            }
        }
        self.save_state(initial_state)
        return initial_state

    def save_state(self, state: Optional[Dict[str, Any]] = None):
        """Atomically saves state to data.json to prevent corruption during unexpected Colab shutdown."""
        if state is not None:
            self.state = state
        self.state["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

        tmp_file = self.config_dir / f"data.json.tmp.{os.getpid()}"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        shutil.move(str(tmp_file), str(self.data_json_path))
        logger.info(f"State synced to {self.data_json_path}")

    def update_version(self, new_version: str):
        """Sets a new version number (e.g. 1.1.0) and updates model_name_tag."""
        self.state["current_version"] = new_version
        self.state["model_name_tag"] = f"SCLLM-v{new_version}"
        self.save_state()
        logger.info(f"Updated project version to: {new_version} (Tag: SCLLM-v{new_version})")

    def register_checkpoint(self, stage: str, step: int, checkpoint_path: str, loss: Optional[float] = None, metrics: Optional[Dict[str, Any]] = None):
        """Records a new checkpoint into state history for restart and continuation."""
        entry = {
            "stage": stage,
            "step": step,
            "checkpoint_path": checkpoint_path,
            "loss": loss,
            "metrics": metrics or {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.state["checkpoints"]["latest_stage"] = stage
        self.state["checkpoints"]["latest_step"] = step
        self.state["checkpoints"]["latest_checkpoint_path"] = checkpoint_path
        self.state["checkpoints"]["can_resume"] = True
        self.state["checkpoints"]["history"].append(entry)
        self.save_state()
        logger.info(f"Checkpoint registered: {checkpoint_path} (Step {step}, Stage {stage})")

    def get_resume_checkpoint(self, stage: Optional[str] = None) -> Optional[str]:
        """Finds the most recent valid checkpoint path to resume an interrupted training run."""
        ck_info = self.state.get("checkpoints", {})
        latest_path = ck_info.get("latest_checkpoint_path")

        if latest_path and os.path.exists(latest_path):
            if stage is None or ck_info.get("latest_stage") == stage:
                return latest_path

        target_dir = self.checkpoints_dir / (stage if stage else "")
        if target_dir.exists():
            checkpoints = [
                d for d in target_dir.glob("**/checkpoint-*")
                if d.is_dir()
            ]
            if checkpoints:
                def extract_step(p):
                    try:
                        return int(p.name.split("-")[-1])
                    except ValueError:
                        return 0
                checkpoints.sort(key=extract_step)
                latest_disk_ck = str(checkpoints[-1])
                logger.info(f"Discovered checkpoint on disk: {latest_disk_ck}")
                return latest_disk_ck

        return None

    def get_stage_adapter(self, stage: str = "stage1_dapt") -> Optional[str]:
        """
        Robustly locates the best available adapter for a given stage:
        1. Checks for final_{stage}_adapter (or final_dapt_adapter / final_sft_adapter).
        2. Falls back to the latest checkpoint-* folder in that stage directory.
        """
        stage_dir = self.checkpoints_dir / stage
        if not stage_dir.exists():
            return None

        # Check standard final save names
        candidates = [
            stage_dir / "final_dapt_adapter",
            stage_dir / "final_sft_adapter",
            stage_dir / f"final_{stage}_adapter"
        ]
        for c in candidates:
            if c.exists() and any(c.iterdir()):
                logger.info(f"Found final adapter for {stage}: {c}")
                return str(c)

        # Fall back to highest step checkpoint-*
        checkpoints = [d for d in stage_dir.glob("checkpoint-*") if d.is_dir() and any(d.iterdir())]
        if checkpoints:
            def extract_step(p):
                try:
                    return int(p.name.split("-")[-1])
                except ValueError:
                    return 0
            checkpoints.sort(key=extract_step)
            best_ck = str(checkpoints[-1])
            logger.info(f"Found latest step checkpoint for {stage}: {best_ck}")
            return best_ck

        return None

    def log_training_run(self, run_details: Dict[str, Any]):
        """Logs completed or stopped training run details."""
        run_details["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        run_details["version"] = self.state.get("current_version", "1.0.0")
        self.state["training_runs"].append(run_details)
        self.save_state()

    def log_evaluation_result(self, test_name: str, score: Optional[float] = None, details: Optional[Dict[str, Any]] = None):
        """Records evaluation benchmark scores into data.json."""
        record = {
            "test_name": test_name,
            "score": score,
            "details": details or {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_version": self.state.get("model_name_tag")
        }
        self.state["evaluation_benchmarks"]["latest_eval_time"] = record["timestamp"]
        self.state["evaluation_benchmarks"]["test_results"].append(record)
        self.save_state()
