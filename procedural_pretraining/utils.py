"""Utility functions for GPT-2 model training.

This module provides helper functions for model creation, checkpointing,
learning rate scheduling, and training infrastructure.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import wandb
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import GPT2Config, GPT2LMHeadModel

logger = logging.getLogger(__name__)

SIZE_TO_CONFIG = {
    "2_4_16": {"n_embd": 16, "n_layer": 2, "n_head": 4},
    "4_4_16": {"n_embd": 16, "n_layer": 4, "n_head": 4},
    "8_4_16": {"n_embd": 16, "n_layer": 8, "n_head": 4},
    "12_4_16": {"n_embd": 16, "n_layer": 12, "n_head": 4},
    "4_4_256": {"n_embd": 256, "n_layer": 4, "n_head": 4},
    "2_4_64": {"n_embd": 64, "n_layer": 2, "n_head": 4},
    "4_8_512": {"n_embd": 512, "n_layer": 4, "n_head": 8},
    "12_1_64": {"n_embd": 64, "n_layer": 12, "n_head": 1},
    "12_3_192": {"n_embd": 192, "n_layer": 12, "n_head": 3},
    "12_6_384": {"n_embd": 384, "n_layer": 12, "n_head": 6},
    "12_12_768": {"n_embd": 768, "n_layer": 12, "n_head": 12},
    "6_12_768": {"n_embd": 768, "n_layer": 6, "n_head": 12},
    "3_12_768": {"n_embd": 768, "n_layer": 3, "n_head": 12},
    "2_12_768": {"n_embd": 768, "n_layer": 2, "n_head": 12},
    "1_12_768": {"n_embd": 768, "n_layer": 1, "n_head": 12},
    "16_8_2048": {"n_embd": 2048, "n_layer": 16, "n_head": 8},
    "24_16_2048": {"n_embd": 2048, "n_layer": 24, "n_head": 16},
}
DEFAULT_DROPOUT = 0.1
MIN_LR_RATIO = 0.1
COSINE_MIDPOINT = 0.5
VALID_DECREASE_MODES = ["cosine", "linear", "constant"]
VAL_SEED_OFFSET = 10000


def set_seed(seed: int) -> None:
    """
    Sets random seeds for reproducibility across libraries.

    Sets the random seed for NumPy, PyTorch CPU, and PyTorch CUDA.

    Args:
        seed: An integer to use as the random seed.

    Raises:
        ValueError: If seed is not a positive integer.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a non-negative integer")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_gpt2_model(
    model_size: str, vocab_size: int, seq_length: int
) -> GPT2LMHeadModel:
    """
    Creates a GPT-2 model with custom configuration.

    Configures a GPT-2 model. The model is configured with appropriate
    embedding dimensions, layer count, and attention heads.

    Args:
        model_size: Configuration key for model size (e.g., "2_4_16" for a
            model with 2 layers, 4 heads, 16-dim embeddings).
        vocab_size: Size of the vocabulary to use.
        seq_length: Maximum sequence length for positional embeddings.

    Returns:
        A GPT-2 model with language modeling head configured for the task.

    Raises:
        ValueError: If the provided model_size is not in the configuration map.
    """
    if model_size not in SIZE_TO_CONFIG:
        valid_sizes = ", ".join(SIZE_TO_CONFIG.keys())
        raise ValueError(
            f"Invalid model_size: {model_size}. Must be one of: {valid_sizes}"
        )

    size_config = SIZE_TO_CONFIG[model_size]

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=seq_length,
        n_ctx=seq_length,
        n_embd=size_config["n_embd"],
        n_layer=size_config["n_layer"],
        n_head=size_config["n_head"],
        resid_pdrop=DEFAULT_DROPOUT,
        embd_pdrop=DEFAULT_DROPOUT,
        attn_pdrop=DEFAULT_DROPOUT,
    )

    model = GPT2LMHeadModel(config)
    return model


def get_lr_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    decrease_mode: str = "cosine",
) -> LambdaLR:
    """
    Creates a learning rate scheduler with warmup and decay.

    Configures a learning rate scheduler that:
    1. Linearly increases the learning rate during warmup steps
    2. Decays the learning rate after warmup according to the specified mode

    Args:
        optimizer: The optimizer whose learning rate will be scheduled.
        warmup_steps: Number of warmup steps with increasing learning rate.
        total_steps: Total number of training steps.
        decrease_mode: The LR decay strategy after warmup.
            Must be one of: "cosine" (cosine decay), "linear" (linear decay),
            or "constant" (no decay - maintains constant LR after warmup).
            Defaults to "cosine".

    Returns:
        A LambdaLR scheduler configured with the specified warmup and decay.

    Raises:
        ValueError: If decrease_mode is not one of the valid options.
    """
    if decrease_mode not in VALID_DECREASE_MODES:
        valid_modes = ", ".join(VALID_DECREASE_MODES)
        raise ValueError(
            f"Invalid decrease_mode: {decrease_mode}. Must be one of: {valid_modes}"
        )

    def lr_lambda(current_step: int) -> float:
        if current_step >= total_steps:
            return MIN_LR_RATIO

        # Linear warmup phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Calculate progress after warmup (0.0 to 1.0)
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )

        # Decay phase - cosine, linear, or constant
        if decrease_mode == "cosine":
            return max(
                MIN_LR_RATIO, COSINE_MIDPOINT * (1.0 + math.cos(math.pi * progress))
            )
        elif decrease_mode == "linear":
            return max(MIN_LR_RATIO, 1.0 - progress)
        else:  # constant
            return 1.0

    return LambdaLR(optimizer, lr_lambda)


def delete_oldest_checkpoint(save_dir: str, save_total_limit: int = 1) -> None:
    """Deletes oldest checkpoints if there are more than the limit.

    Args:
        save_dir: Directory containing checkpoint files.
        save_total_limit: Maximum number of checkpoints to keep per prefix.
    """
    for prefix in ["pytorch_model_", "best_pytorch_model_"]:
        try:
            checkpoints = [
                f
                for f in os.listdir(save_dir)
                if f.startswith(prefix) and f.endswith(".pth")
            ]

            checkpoints.sort(key=lambda x: sort_checkpoint(x))

            if len(checkpoints) > save_total_limit:
                for i in range(len(checkpoints) - save_total_limit):
                    oldest_checkpoint = checkpoints[i]
                    oldest_path = os.path.join(save_dir, oldest_checkpoint)
                    print(f"Remove checkpoint: {oldest_path}")
                    os.remove(oldest_path)
        except Exception as e:
            print(f"Warning: Error while cleaning up checkpoints: {str(e)}")


def save_checkpoint(
    epoch,
    step,
    model,
    optimizer,
    scheduler,
    val_loss,
    val_acc,
    save_dir,
    tokenizer=None,
    is_best=False,
    filename=None,
):
    """
    Save model checkpoint.

    Args:
        epoch: Current epoch number
        step: Current step number
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Scheduler to save
        val_loss: Current validation loss
        val_acc: Current validation accuracy
        save_dir: Directory to save the checkpoint
        tokenizer: Optional tokenizer state to persist
        is_best: Whether this is the best model so far based on validation accuracy
        filename: Optional custom filename (overrides default naming)
    """
    save_dir = os.fspath(save_dir)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }
    if tokenizer is not None:
        ckpt["tokenizer"] = tokenizer

    if filename is not None:
        ckpt_path = os.path.join(save_dir, filename)
    elif is_best:
        ckpt_path = os.path.join(save_dir, f"best_pytorch_model_{epoch}_step{step}.pth")
    else:
        ckpt_path = os.path.join(save_dir, f"pytorch_model_{epoch}_step{step}.pth")

    torch.save(ckpt, ckpt_path)


def sort_checkpoint(ck_name: str) -> Tuple[int, int]:
    """Extracts epoch and step numbers from checkpoint name for sorting.

    Args:
        ck_name: Checkpoint filename.

    Returns:
        Tuple of (epoch, step) for sorting.
    """
    try:
        # For filenames like pytorch_model_1_step100.pth
        if "step" in ck_name:
            parts = ck_name.replace(".pth", "").split("_")
            epoch = int(parts[-2])
            step = int(parts[-1].replace("step", ""))
            return epoch, step
        else:
            parts = ck_name.replace(".pth", "").split("_")
            epoch = int(parts[-2])
            step = int(parts[-1])
            return epoch, step
    except (ValueError, IndexError):
        print(f"Warning: Could not parse checkpoint filename: {ck_name}")
        return -1, -1


def load_last_checkpoint(model, optimizer, scheduler, save_dir, tokenizer=None):
    """
    Load the last checkpoint.

    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        save_dir: Directory containing checkpoints

    Returns:
        tuple: (model, optimizer, scheduler, epoch, step, best_val_acc)
    """
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith("pytorch_model_")]
    if len(checkpoints) == 0:
        raise ValueError("No checkpoint found")

    checkpoints.sort(key=lambda x: sort_checkpoint(x))
    sd = torch.load(os.path.join(save_dir, checkpoints[-1]))

    epoch = sd["epoch"]
    step = sd["step"]

    best_val_acc = sd.get("val_acc", 0.0)

    model.load_state_dict(sd["model"])
    optimizer.load_state_dict(sd["optimizer"])
    scheduler.load_state_dict(sd["scheduler"])
    if tokenizer is not None and "tokenizer" in sd:
        tokenizer.load_state_dict(sd["tokenizer"])
        return model, optimizer, scheduler, epoch, step, best_val_acc, tokenizer
    else:
        return model, optimizer, scheduler, epoch, step, best_val_acc


def wandb_log(data: Dict[str, Any], args: Any) -> None:
    """Logs data to Weights & Biases if enabled.

    Args:
        data: Dictionary of data to log.
        args: Arguments object with wandb_enable flag.
    """
    if hasattr(args, "wandb_enable") and args.wandb_enable:
        wandb.log(data)
    else:
        print(data)


def args_post_init(args: Any, task_name: str) -> Any:
    """Post-initializes arguments with derived values.

    Args:
        args: Arguments object containing initial configuration.
        task_name: Name of the task (e.g., 'stack', 'set', 'dyck').

    Returns:
        Updated arguments object with derived values.
    """
    # Set device
    args.device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )

    # Set wandb name only if not provided (respect YAML/CLI)
    if not hasattr(args, "wandb_name") or args.wandb_name in [None, ""]:
        wandb_name = f"{task_name}_learning"

        # Add key parameters to the name
        if hasattr(args, "seq_length"):
            wandb_name += f"_seq{args.seq_length}"
        if hasattr(args, "vocab_size"):
            wandb_name += f"_vocab{args.vocab_size}"
        if hasattr(args, "gpt2_size"):
            wandb_name += f"_{args.gpt2_size}"

        args.wandb_name = wandb_name

    # Set paths
    if not hasattr(args, "run_name") or args.run_name is None:
        # Create a default run name based on parameters
        run_name = f"{task_name}"
        if hasattr(args, "seq_length"):
            run_name += f"-{args.seq_length}"
        if hasattr(args, "gpt2_size"):
            run_name += f"-{args.gpt2_size}"
        if hasattr(args, "max_steps"):
            run_name += f"-{args.max_steps}steps"
        args.run_name = run_name
    else:
        args.wandb_name += f"_{args.run_name}"

    # Create save directory
    args.save_dir = os.path.join(args.save_dir, args.run_name)

    metrics_filename = "metrics"
    if hasattr(args, "seq_length"):
        metrics_filename += f"_{args.seq_length}"
    if hasattr(args, "gpt2_size"):
        metrics_filename += f"_{args.gpt2_size}"
    metrics_filename += ".json"

    args.metrics_path = os.path.join(args.save_dir, metrics_filename)

    return args


def ensure_dir_exists(dir_path: str) -> None:
    """Ensures that a directory exists, creating it if necessary.

    Args:
        dir_path: Path to the directory.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def load_config_file(path: str) -> dict[str, Any]:
    """Loads configuration from a YAML or JSON file.

    Args:
        path: Path to the configuration file.

    Returns:
        Dictionary containing the configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ImportError: If PyYAML is needed but not installed.
        ValueError: If the file extension is not supported.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    if p.suffix.lower() in [".yml", ".yaml"]:
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML is not installed. `pip install pyyaml`") from e
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    elif p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config extension: {p.suffix}")


def merge_config_with_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Merges YAML/JSON config file with command line arguments.

    CLI args override config values, config values override argparse defaults.

    Args:
        parser: ArgumentParser instance with defined arguments.

    Returns:
        Namespace containing merged configuration values.
    """
    args = parser.parse_args()

    if not getattr(args, "config", None):
        return args

    config = load_config_file(args.config)

    defaults = {
        action.dest: action.default
        for action in parser._actions
        if hasattr(action, "dest") and action.dest != "help"
    }

    args_dict = vars(args)
    for key, value in config.items():
        if key in args_dict and args_dict[key] == defaults.get(key):
            args_dict[key] = value
        elif key not in args_dict:
            logger.warning("Config key '%s' not recognized", key)

    return argparse.Namespace(**args_dict)
