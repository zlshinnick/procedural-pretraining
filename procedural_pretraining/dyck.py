"""Pretraining script for GPT-2 models on Dyck language datasets.

This module provides training functionality for Dyck language tasks.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from typing import Any

import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW, Optimizer
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

_VAL_EPOCH_SEED = 1000


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
            - max_steps: Maximum training steps (None for no limit).
            - gradient_accumulation_steps: Steps to accumulate before update
              (default 1).
            Also reads/writes state attributes: global_step, best_val_acc,
            patience_counter.
        val_loader: DataLoader containing validation data.

    Returns:
        List of training loss values for each optimizer step.
    """
    model.train()

    train_loss_history = []
    global_step = args.global_step if hasattr(args, "global_step") else 0
    patience_counter = args.patience_counter if hasattr(args, "patience_counter") else 0
    best_val_acc = args.best_val_acc if hasattr(args, "best_val_acc") else 0.0

    gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", 1)
    accumulation_counter = 0
    accumulated_loss = 0.0
    accumulated_correct = 0
    accumulated_tokens = 0

    for step, batch in enumerate(tqdm(train_loader, desc="Training")):
        input_ids = batch["input_ids"].to(args.device)
        targets = batch["targets"].to(args.device)
        loss_mask = batch["loss_mask"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        loss_mask_flat = loss_mask.reshape(-1)

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(logits_flat, targets_flat)

        masked_losses = losses * loss_mask_flat
        num_valid_tokens = loss_mask_flat.sum()
        if num_valid_tokens > 0:
            loss = masked_losses.sum() / num_valid_tokens
        else:
            loss = torch.tensor(0.0, requires_grad=True, device=args.device)

        scaled_loss = loss / gradient_accumulation_steps
        scaled_loss.backward()
        accumulated_loss += loss.item()

        with torch.no_grad():
            predictions = torch.argmax(logits_flat, dim=-1)
            correct = ((predictions == targets_flat) * loss_mask_flat).sum().item()
            accumulated_correct += correct
            accumulated_tokens += num_valid_tokens.item()

        accumulation_counter += 1

        if accumulation_counter == gradient_accumulation_steps:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            avg_loss = accumulated_loss / gradient_accumulation_steps
            train_acc = (
                accumulated_correct / accumulated_tokens
                if accumulated_tokens > 0
                else 0.0
            )
            train_loss_history.append(avg_loss)
            wandb_log(
                {
                    "train_loss": avg_loss,
                    "train_accuracy": train_acc,
                    "lr": scheduler.get_last_lr()[0],
                    "global_step": global_step,
                },
                args,
            )

            logger.info(
                "Step %d: loss=%.4f, acc=%.4f, lr=%.2e",
                global_step,
                avg_loss,
                train_acc,
                scheduler.get_last_lr()[0],
            )

            accumulated_loss = 0.0
            accumulated_correct = 0
            accumulated_tokens = 0
            accumulation_counter = 0
            global_step += 1
        else:
            continue

        if global_step % args.save_every_steps == 0:
            logger.info("--- Running validation at step %d ---", global_step)
            val_loss, val_acc = validation(model, val_loader, args)

            wandb_log(
                {
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "global_step": global_step,
                },
                args,
            )

            logger.info(
                "Step %d: Val Loss = %.6f, Val Accuracy = %.6f",
                global_step,
                val_loss,
                val_acc,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                save_checkpoint(
                    epoch=1,
                    step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    save_dir=args.save_dir,
                    is_best=True,
                )
                logger.info("New best model with val_accuracy: %.6f", best_val_acc)
            else:
                patience_counter += 1

            save_checkpoint(
                epoch=1,
                step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                val_loss=val_loss,
                val_acc=val_acc,
                save_dir=args.save_dir,
            )
            delete_oldest_checkpoint(args.save_dir, args.save_total_limit)

            model.train()

        if args.max_steps is not None and global_step >= args.max_steps:
            logger.info(
                "Reached maximum steps (%d). Stopping training.", args.max_steps
            )
            break

    args.global_step = global_step
    args.best_val_acc = best_val_acc
    args.patience_counter = patience_counter

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
        args: Configuration object containing device and other parameters.

    Returns:
        A tuple containing:
            - avg_loss: Average loss per token across the validation set.
            - accuracy: Token-level accuracy across the validation set.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(args.device)
            targets = batch["targets"].to(args.device)
            loss_mask = batch["loss_mask"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            loss_mask_flat = loss_mask.reshape(-1)

            loss_fct = nn.CrossEntropyLoss(reduction="none")
            losses = loss_fct(logits_flat, targets_flat)

            masked_losses = losses * loss_mask_flat
            valid_tokens = loss_mask_flat.sum().item()

            if valid_tokens > 0:
                batch_loss = masked_losses.sum().item()
                total_loss += batch_loss

                predictions = torch.argmax(logits_flat, dim=-1)
                correct_predictions = (predictions == targets_flat) * loss_mask_flat
                total_correct += correct_predictions.sum().item()
                total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    return avg_loss, accuracy


def main(args: Any) -> None:
    """Main training function for Dyck language model pretraining.

    Args:
        args: Configuration object containing all training parameters including
            task type, model configuration, and training hyperparameters.
    """
    args = args_post_init(args, args.task)
    set_seed(args.seed)

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

    vocab_size = args.k * 2 + 1

    model = create_gpt2_model(args.gpt2_size, vocab_size, args.seq_length)
    logger.info("Model config: %s", model.config)
    model.to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total parameters: %d", total_params)

    gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", 1)
    effective_batch_size = args.batch_size * gradient_accumulation_steps
    logger.info(
        "Batch size: %d, Gradient accumulation steps: %d",
        args.batch_size,
        gradient_accumulation_steps,
    )
    logger.info("Effective batch size: %d", effective_batch_size)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = (
        args.max_steps if args.max_steps else args.epochs * args.steps_per_epoch
    )
    warmup_steps = getattr(args, "warmup_steps", 0)
    lr_schedule = getattr(args, "lr_schedule", "constant")

    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        decrease_mode=lr_schedule,
    )
    logger.info(
        "LR schedule: %s, warmup_steps: %d, total_steps: %d",
        lr_schedule,
        warmup_steps,
        total_steps,
    )

    args.best_val_loss = float("inf")
    args.global_step = 0
    start_epoch = 0
    args.best_val_acc = 0.0
    args.patience_counter = 0

    if args.resume:
        (
            model,
            optimizer,
            scheduler,
            start_epoch,
            args.global_step,
            args.best_val_loss,
        ) = load_last_checkpoint(model, optimizer, scheduler, args.save_dir)
        logger.info("Resuming from epoch %d, step %d", start_epoch, args.global_step)

    all_metrics = {
        "train_loss_history": [],
        "val_metrics": [],
    }

    task_modules = {
        "dyck": "procedural_data.dyck",
        "dyck_shuffle": "procedural_data.dyck_shuffle",
    }
    if args.task not in task_modules:
        raise ValueError(f"Unknown task: {args.task}. Must be 'dyck' or 'dyck_shuffle'")
    build_dataloader = importlib.import_module(task_modules[args.task]).build_dataloader

    _, val_loader = build_dataloader(args, epoch_seed=_VAL_EPOCH_SEED)

    for epoch in range(start_epoch, args.epochs):
        logger.info(
            "==================== Epoch: %d / %d ===========================",
            epoch + 1,
            args.epochs,
        )

        train_loader, _ = build_dataloader(args, epoch_seed=epoch)

        train_loss_history = train_epoch(
            model, train_loader, optimizer, scheduler, args, val_loader
        )
        all_metrics["train_loss_history"].append({epoch: train_loss_history})

        if hasattr(args, "patience_counter") and args.patience_counter >= args.patience:
            logger.info(
                "Early stopping: no improvement in val accuracy for %d validations.",
                args.patience,
            )
            break

        with open(args.metrics_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)

        if args.max_steps is not None and args.global_step >= args.max_steps:
            logger.info(
                "Reached maximum steps (%d). Stopping training.", args.max_steps
            )
            break

    save_checkpoint(
        epoch + 1,
        args.global_step,
        model,
        optimizer,
        scheduler,
        args.best_val_loss,
        args.best_val_acc,
        args.save_dir,
    )

    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    if args.wandb_enable:
        wandb.finish()


def create_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser for Dyck language training.

    Returns:
        Configured ArgumentParser with all training options.
    """
    parser = argparse.ArgumentParser(
        description="Train a GPT-2 model on Dyck language sequences"
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
            "4_4_16",
            "8_4_16",
            "12_4_16",
            "4_4_256",
            "2_4_64",
            "4_8_512",
            "12_1_64",
            "12_3_192",
            "12_6_384",
            "12_12_768",
            "6_12_768",
            "3_12_768",
            "2_12_768",
            "1_12_768",
            "16_8_2048",
            "24_16_2048",
        ],
        help="Size of the GPT-2 model (format: layers_heads_embedding)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="dyck",
        choices=["dyck", "dyck_shuffle"],
        help="Synthetic task to train on",
    )

    # Dyck sequence parameters
    parser.add_argument(
        "--k", type=int, default=16, help="Number of bracket types (k in Dyck-k)"
    )
    parser.add_argument(
        "--p_open", type=float, default=0.5, help="Probability of opening a bracket"
    )
    parser.add_argument(
        "--max_depth", type=int, default=16, help="Maximum nesting depth"
    )
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--steps_per_epoch", type=int, default=1000, help="Steps per epoch"
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=0, help="LR warmup steps")
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="constant",
        choices=["cosine", "linear", "constant"],
        help="LR schedule after warmup",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Max gradient norm"
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Max epochs")
    parser.add_argument("--max_steps", type=int, default=1000000, help="Max steps")

    # System parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")

    # Checkpointing
    parser.add_argument("--save_dir", type=str, help="Checkpoint directory")
    parser.add_argument(
        "--save_every_steps", type=int, default=100, help="Checkpoint frequency"
    )
    parser.add_argument(
        "--save_total_limit", type=int, default=1, help="Max checkpoints to keep"
    )
    parser.add_argument(
        "--patience", type=int, default=1000001, help="Early stopping patience"
    )

    # Logging
    parser.add_argument(
        "--wandb_project", type=str, default="dyck_pretraining", help="W&B project"
    )
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb_enable", action="store_true", help="Enable W&B")

    # Resume
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--resume_run_id", type=str, default=None, help="W&B run ID")

    return parser


if __name__ == "__main__":
    args = merge_config_with_args(create_parser())
    main(args)
