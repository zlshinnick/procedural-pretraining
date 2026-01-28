"""Pretraining script for GPT-2 models on algorithmic sequence tasks.

This module provides training functionality for various algorithmic tasks.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import wandb
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import (
    args_post_init,
    create_gpt2_model,
    delete_oldest_checkpoint,
    ensure_dir_exists,
    get_lr_scheduler,
    load_last_checkpoint,
    merge_config_with_args,
    save_checkpoint,
    set_seed,
    wandb_log,
)

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    args: Any,
    val_loader: DataLoader,
) -> list[float]:
    """Trains the model for one epoch.

    Args:
        model: GPT2 model to train.
        train_loader: DataLoader containing training data.
        optimizer: Optimizer for updating model parameters.
        scheduler: Learning rate scheduler.
        args: Configuration object with required attributes:
            - device: Device to run training on.
            - max_grad_norm: Maximum gradient norm for clipping.
            - save_every_steps: Frequency of validation and checkpointing.
            - save_dir: Directory to save checkpoints.
            - save_total_limit: Maximum number of checkpoints to keep.
            - patience: Validation checks without improvement before early
              stopping (0 to disable).
            - max_steps: Maximum training steps (None for no limit).
            Also reads/writes state attributes: global_step, best_val_acc,
            no_improve_count.
        val_loader: DataLoader containing validation data.

    Returns:
        List of training loss values for each training step.
    """
    model.train()

    train_loss_history = []
    global_step = args.global_step if hasattr(args, "global_step") else 0
    best_val_acc = args.best_val_acc if hasattr(args, "best_val_acc") else 0.0
    no_improve_count = args.no_improve_count if hasattr(args, "no_improve_count") else 0

    for step, batch in enumerate(tqdm(train_loader, desc="Training")):
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        targets = batch["targets"].to(args.device)
        loss_mask = batch["loss_mask"].to(args.device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        masked_targets = targets.clone()
        masked_targets[loss_mask == 0] = -100

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), masked_targets.view(-1))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        train_loss_history.append(loss.item())

        log_dict = {
            "train_loss": loss.item(),
            "lr": scheduler.get_last_lr()[0],
            "global_step": global_step,
        }
        wandb_log(log_dict, args)

        global_step += 1

        if global_step % args.save_every_steps == 0:
            val_loss, val_acc = validation(model, val_loader, args)

            wandb_log(
                {
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "global_step": global_step,
                },
                args,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve_count = 0
            else:
                no_improve_count += 1

            save_checkpoint(
                1,
                global_step,
                model,
                optimizer,
                scheduler,
                val_loss,
                val_acc,
                args.save_dir,
            )
            delete_oldest_checkpoint(args.save_dir, args.save_total_limit)

            if args.patience > 0 and no_improve_count >= args.patience:
                args.global_step = global_step
                args.best_val_acc = best_val_acc
                args.no_improve_count = no_improve_count
                return train_loss_history

            model.train()

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    args.global_step = global_step
    args.best_val_acc = best_val_acc
    args.no_improve_count = no_improve_count

    return train_loss_history


def validation(
    model: nn.Module,
    val_loader: DataLoader,
    args: Any,
) -> tuple[float, float]:
    """Evaluates the model on the validation set.

    Args:
        model: GPT2 model to evaluate.
        val_loader: DataLoader containing validation data.
        args: Configuration object with required attributes:

    Returns:
        A tuple containing:
            - avg_loss: Average loss per token across the validation set.
            - accuracy: Token-level accuracy across the validation set.
    """
    device = args.device
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            masked_targets = targets.clone()
            masked_targets[loss_mask == 0] = -100

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.reshape(-1, logits.size(-1)), masked_targets.reshape(-1)
            )

            predictions = torch.argmax(logits, dim=-1)
            valid_positions = loss_mask == 1
            correct = ((predictions == targets) & valid_positions).sum().item()
            total_positions = valid_positions.sum().item()

            total_loss += loss.item() * input_ids.size(0)
            total_correct += correct
            total_samples += total_positions

    if len(val_loader.dataset) > 0:
        avg_loss = total_loss / len(val_loader.dataset)
    else:
        avg_loss = float("nan")

    if total_samples > 0:
        accuracy = total_correct / total_samples
    else:
        accuracy = 0.0

    return avg_loss, accuracy


def main(args: argparse.Namespace) -> None:
    """Main training function for algorithmic task pretraining.

    Args:
        args: Configuration object containing all training parameters including
            task type, model configuration, and training hyperparameters.
    """
    args = args_post_init(args, task_name=args.task)
    args.separator_token = args.vocab_size
    args.pad_token = args.vocab_size + 1

    set_seed(args.seed)

    if getattr(args, "wandb_name", None) in [None, ""]:
        default_name = getattr(args, "run_name", None)
        if not default_name:
            default_name = f"{args.task}_len{args.seq_length}_vocab_{args.vocab_size}"
            default_name += f"_seed{args.seed}"
        args.wandb_name = default_name

    task_modules = {
        "stack": "procedural_data.stack",
        "set": "procedural_data.set",
        "identity": "procedural_data.identity",
        "reverse": "procedural_data.reverse",
        "union": "procedural_data.union",
        "delete": "procedural_data.delete",
        "sort": "procedural_data.sort",
    }
    if args.task not in task_modules:
        raise ValueError(f"Unsupported task: {args.task}")
    build_dataloader = importlib.import_module(task_modules[args.task]).build_dataloader

    if args.wandb_enable:
        wandb_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_name,
            "config": args.__dict__,
        }
        if args.resume_run_id is not None:
            wandb_kwargs.update(id=args.resume_run_id, resume="must")
        wandb.init(**wandb_kwargs)

    ensure_dir_exists(args.save_dir)

    if getattr(args, "config", None):
        try:
            src_cfg = Path(args.config)
            if src_cfg.exists():
                dst_cfg = Path(args.save_dir) / src_cfg.name
                if dst_cfg.exists():
                    dst_cfg = Path(args.save_dir) / f"config_used{src_cfg.suffix}"
                shutil.copy2(src_cfg, dst_cfg)
        except Exception as e:
            logger.warning(f"Failed to copy config file to save_dir: {e}")

    if args.task == "union":
        n_pos = args.seq_length * 2 + 2
    elif args.task == "delete":
        n_pos = args.seq_length * 2 + 3
    else:
        n_pos = args.seq_length * 2 + 1
    vocab_for_model = args.vocab_size + 2  # separator + pad tokens
    model = create_gpt2_model(args.gpt2_size, vocab_for_model, n_pos)

    logger.info(str(model.config))
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    model.to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        decrease_mode=args.lr_schedule,
    )

    args.best_val_acc = 0.0
    args.no_improve_count = 0
    args.global_step = 0
    start_epoch = 0

    if args.resume:
        try:
            model, optimizer, scheduler, start_epoch, args.global_step, _ = (
                load_last_checkpoint(model, optimizer, scheduler, args.save_dir)
            )
            logger.info(f"Resuming from epoch {start_epoch}, step {args.global_step}")
            args.no_improve_count = 0
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {e}. Starting from scratch.")

    all_metrics = {
        "train_loss_history": [],
        "val_metrics": [],
        "best_val_acc": args.best_val_acc,
    }

    for epoch in range(start_epoch, args.epochs):
        logger.info(
            f"==================== Epoch: {epoch + 1} / {args.epochs} ==========================="
        )

        if args.use_curriculum:
            args.curriculum_start = getattr(args, "curriculum_start", 2)
            args.curriculum_step_size = getattr(args, "curriculum_step_size", 2)
            args.curriculum_seq_length = getattr(
                args, "curriculum_seq_length", args.curriculum_start
            )
            args.train_samples = args.batch_size * args.steps_per_epoch
            args.val_samples = args.batch_size * 10
            current_seq_length = args.curriculum_seq_length
            logger.info(
                f"[Curriculum] Epoch {epoch} using seq_length = {current_seq_length}"
            )
        else:
            current_seq_length = args.seq_length

        train_loader, val_loader = build_dataloader(
            args, epoch_seed=epoch, seq_length=current_seq_length
        )

        train_loss_history = train_epoch(
            model, train_loader, optimizer, scheduler, args, val_loader
        )
        all_metrics["train_loss_history"].append({epoch: train_loss_history})
        all_metrics["best_val_acc"] = args.best_val_acc

        with open(args.metrics_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)

        if (
            args.use_curriculum
            and args.best_val_acc >= getattr(args, "curriculum_mastery_threshold", 0.99)
            and args.curriculum_seq_length < args.seq_length
        ):
            args.curriculum_seq_length = min(
                args.curriculum_seq_length + args.curriculum_step_size, args.seq_length
            )
            logger.info(
                f"[Curriculum] Mastery reached, increasing seq_length to {args.curriculum_seq_length}"
            )
            args.best_val_acc = 0.0

        if args.max_steps is not None and args.global_step >= args.max_steps:
            logger.info(f"Reached maximum steps ({args.max_steps}). Stopping training.")
            break

        if args.patience > 0 and args.no_improve_count >= args.patience:
            logger.info(
                f"Early stopping triggered. No improvement in validation accuracy for {args.no_improve_count} epochs."
            )
            break

    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"Training completed. Best accuracy: {args.best_val_acc:.6f}")

    if args.wandb_enable:
        wandb.finish()


def create_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser for algorithmic task training.

    Returns:
        Configured ArgumentParser with all training options.
    """
    parser = argparse.ArgumentParser(
        description="Train a GPT-2 model on algorithmic sequence tasks"
    )

    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML/JSON config"
    )

    # Model configuration
    parser.add_argument(
        "--gpt2_size",
        type=str,
        default="2_4_16",
        choices=[
            "2_4_16",
            "12_12_768",
            "6_12_768",
            "3_12_768",
            "2_12_768",
            "1_12_768",
            "4_4_256",
            "4_8_512",
            "12_3_192",
            "4_4_16",
            "8_4_16",
            "12_4_16",
            "16_8_2048",
        ],
        help="Model size (format: layers_heads_embedding)",
    )
    # Dataset configuration
    parser.add_argument("--vocab_size", type=int, default=21, help="Vocabulary size")
    parser.add_argument(
        "--task",
        type=str,
        default="stack",
        choices=["stack", "set", "identity", "reverse", "union", "delete", "sort"],
        help="Algorithmic task to train on",
    )
    parser.add_argument("--seq_length", type=int, default=20, help="Sequence length")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--steps_per_epoch", type=int, default=100, help="Steps per epoch"
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--warmup_steps", type=int, default=1000, help="LR warmup steps"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Max gradient norm"
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "constant"],
        help="LR schedule after warmup",
    )
    parser.add_argument("--epochs", type=int, default=10000, help="Max epochs")
    parser.add_argument("--max_steps", type=int, default=20000, help="Max steps")

    # System parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Dataloader workers"
    )

    # Checkpointing
    parser.add_argument("--save_dir", type=str, help="Checkpoint directory")
    parser.add_argument(
        "--save_every_steps", type=int, default=10, help="Checkpoint frequency"
    )
    parser.add_argument(
        "--save_total_limit", type=int, default=1, help="Max checkpoints to keep"
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="Early stopping patience"
    )

    # Logging
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="algorithmic_pretraining",
        help="W&B project",
    )
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb_enable", action="store_true", help="Enable W&B")
    parser.add_argument("--run_name", type=str, default=None, help="Run name")

    # Resume
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--resume_run_id", type=str, default=None, help="W&B run ID")

    # Curriculum learning
    parser.add_argument(
        "--use_curriculum", action="store_true", help="Enable curriculum"
    )
    parser.add_argument(
        "--curriculum_start", type=int, default=2, help="Initial seq length"
    )
    parser.add_argument(
        "--curriculum_step_size", type=int, default=2, help="Seq length increment"
    )
    parser.add_argument(
        "--curriculum_step_epochs", type=int, default=10, help="Epochs per increment"
    )
    parser.add_argument(
        "--curriculum_mastery_threshold",
        type=float,
        default=0.99,
        help="Accuracy to advance curriculum",
    )

    return parser


if __name__ == "__main__":
    args = merge_config_with_args(create_parser())
    main(args)
