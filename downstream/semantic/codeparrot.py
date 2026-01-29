"""
    Pretrain a GPT-2 model on the CodeParrot dataset using 🤗 Accelerate.
    https://huggingface.co/codeparrot/codeparrot-small/blob/main/codeparrot_training.py
    https://huggingface.co/codeparrot/codeparrot-small

    Example usage:
    # Using a config file:
    accelerate launch downstream/semantic/codeparrot.py --config configs/codeparrot.yaml

    # Using command-line arguments (overrides config file values):
    accelerate launch downstream/semantic/codeparrot.py \
        --config configs/codeparrot.yaml \
        --max_train_steps 10000

    # Using only command-line arguments:
    accelerate launch downstream/semantic/codeparrot.py \
        --wandb_enable \
        --dataset_name_train codeparrot/codeparrot-clean-train \
        --dataset_name_valid codeparrot/codeparrot-clean-valid \
        --pretrained_tokenizer codeparrot/codeparrot \
        --pretrained_model <...> \
        --output_dir <...>
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import logging
import json
from typing import Any, Dict, List, Optional, Tuple

import yaml

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    get_scheduler,
    set_seed,
)
from datasets import load_dataset

from downstream.utils import initialize_model
from downstream.semantic.data.codeparrot_dataset import ConstantLengthDataset

logger = logging.getLogger(__name__)


def get_grouped_params(
    model: torch.nn.Module,
    args: argparse.Namespace,
    no_decay: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Get parameter groups for optimizer with and without weight decay.

    Args:
        model: The model to get parameters from.
        args: Arguments containing weight_decay value.
        no_decay: List of parameter name substrings that should not have weight decay.
            Defaults to ["bias", "LayerNorm.weight"].

    Returns:
        List of parameter group dicts for the optimizer.
    """
    if no_decay is None:
        no_decay = ["bias", "LayerNorm.weight"]
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": args.weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    optimizer: AdamW,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    completed_steps: int,
    eval_metrics: Dict[str, float],
    args: argparse.Namespace,
    checkpoint_type: str = "latest",
) -> None:
    """
    Save a checkpoint with all necessary components for resuming training.

    Args:
        accelerator: The Accelerator instance.
        model: The model to save.
        tokenizer: The tokenizer to save.
        optimizer: The optimizer whose state to save.
        lr_scheduler: The learning rate scheduler whose state to save.
        completed_steps: Number of training steps completed.
        eval_metrics: Dictionary of evaluation metrics.
        args: Training arguments.
        checkpoint_type: "latest" or "best" - determines the subdirectory name.
    """
    accelerator.wait_for_everyone()

    # Determine checkpoint path
    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{checkpoint_type}")

    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model and tokenizer
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(checkpoint_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        training_state = {
            "completed_steps": completed_steps,
            "eval_metrics": eval_metrics,
            "wandb_run_id": wandb.run.id if wandb.run is not None else None,
            "best_eval_perplexity": args.best_eval_perplexity,
            "args": vars(args),
        }

        with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
            json.dump(training_state, f, indent=2)

    # Save optimizer and scheduler states (accelerate handles distributed saving)
    accelerator.save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state()
            if torch.cuda.is_available()
            else None,
        },
        os.path.join(checkpoint_dir, "training_states.pt"),
    )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info(f"Checkpoint saved to {checkpoint_dir}")


def load_checkpoint(
    accelerator: Accelerator,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint if it exists and return resume information.

    Args:
        accelerator: The Accelerator instance.
        args: Training arguments containing output_dir.

    Returns:
        Dict with resume information or None if no checkpoint found.
    """
    checkpoint_dir = os.path.join(args.output_dir, "checkpoint-latest")

    if not os.path.exists(checkpoint_dir):
        return None

    # Load training state
    training_state_path = os.path.join(checkpoint_dir, "training_state.json")
    if not os.path.exists(training_state_path):
        logger.warning("Checkpoint directory exists but no training_state.json found")
        return None

    with open(training_state_path, "r") as f:
        training_state = json.load(f)

    # Load optimizer and scheduler states
    training_states_path = os.path.join(checkpoint_dir, "training_states.pt")
    if not os.path.exists(training_states_path):
        logger.warning("Checkpoint directory exists but no training_states.pt found")
        return None

    checkpoint_data = torch.load(training_states_path, map_location="cpu")

    resume_info = {
        "checkpoint_dir": checkpoint_dir,
        "completed_steps": training_state["completed_steps"],
        "eval_metrics": training_state["eval_metrics"],
        "wandb_run_id": training_state.get("wandb_run_id"),
        "best_eval_perplexity": training_state.get(
            "best_eval_perplexity", float("inf")
        ),
        "optimizer_state_dict": checkpoint_data["optimizer_state_dict"],
        "lr_scheduler_state_dict": checkpoint_data["lr_scheduler_state_dict"],
        "random_state": checkpoint_data.get("random_state"),
        "cuda_random_state": checkpoint_data.get("cuda_random_state"),
    }

    return resume_info


def create_dataloaders(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and evaluation dataloaders.

    Args:
        args: Training arguments containing dataset names and batch sizes.
        tokenizer: The tokenizer for the ConstantLengthDataset.

    Returns:
        Tuple of (train_dataloader, eval_dataloader).
    """
    train_data = load_dataset(args.dataset_name_train, split="train", streaming=True)
    train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    valid_data = load_dataset(args.dataset_name_valid, split="train", streaming=True)

    train_dataset = ConstantLengthDataset(
        tokenizer, train_data, infinite=True, seq_length=args.seq_length
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer, valid_data, infinite=False, seq_length=args.seq_length
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    eval_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    return train_dataloader, eval_dataloader


def evaluate(
    args: argparse.Namespace,
    model: torch.nn.Module,
    eval_dataloader: DataLoader,
    accelerator: Accelerator,
) -> Tuple[float, float]:
    """
    Evaluate the model on the validation set.

    Args:
        args: Training arguments containing max_eval_steps and valid_batch_size.
        model: The model to evaluate.
        eval_dataloader: DataLoader for evaluation data.
        accelerator: The Accelerator instance for distributed gathering.

    Returns:
        Tuple of (loss, perplexity).
    """
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(args.valid_batch_size)
        losses.append(accelerator.gather(loss))
        if args.max_eval_steps > 0 and step >= args.max_eval_steps:
            break
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()


def log_metrics(
    accelerator: Accelerator,
    step: int,
    metrics: Dict[str, float],
    split: str = "train",
) -> None:
    """
    Log metrics to console and wandb.

    Args:
        accelerator: The Accelerator instance.
        step: Current training step.
        metrics: Dictionary of metrics to log.
        split: Data split name ("train" or "eval").
    """
    if accelerator.is_main_process:
        logger.info(f"Step {step}: {metrics}")

        # Log to wandb if enabled
        wandb_metrics = {f"{split}/{k}": v for k, v in metrics.items()}
        wandb_metrics["step"] = step
        if wandb.run is not None:
            wandb.log(wandb_metrics)


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.wandb_enable else None,
        project_dir=args.output_dir if args.wandb_enable else None,
    )

    # Calculate effective batch size
    samples_per_step = (
        accelerator.state.num_processes
        * args.train_batch_size
        * args.gradient_accumulation_steps
    )
    args.samples_per_step = samples_per_step

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.info(f"Samples per step: {samples_per_step}")

    # Set seed
    set_seed(args.seed)

    # Check for existing checkpoint
    resume_info = None
    if accelerator.is_main_process:
        if os.path.exists(args.output_dir):
            resume_info = load_checkpoint(accelerator, args)
            if resume_info:
                logger.info(
                    f"Found checkpoint at step {resume_info['completed_steps']}, will resume training"
                )
            else:
                # If output dir exists but no valid checkpoint, check if it's empty
                if os.listdir(args.output_dir):
                    # Allow continuation if there are only log files or other non-critical files
                    checkpoint_dirs = [
                        d
                        for d in os.listdir(args.output_dir)
                        if d.startswith("checkpoint-")
                    ]
                    if checkpoint_dirs:
                        raise ValueError(
                            f"Output directory {args.output_dir} contains incomplete checkpoints. Please clean or use a different directory."
                        )
        else:
            os.makedirs(args.output_dir, exist_ok=False)

    # Broadcast resume info to all processes
    resume_info = broadcast_object_list([resume_info])[0]

    # Setup wandb
    if accelerator.is_main_process and args.wandb_enable:
        wandb_id = resume_info["wandb_run_id"] if resume_info else None
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            resume="must" if wandb_id else None,
            id=wandb_id,
        )

    # Initialize tracking variables
    args.best_eval_perplexity = (
        resume_info["best_eval_perplexity"] if resume_info else float("inf")
    )

    # Setup tokenizer
    logger.info(f"Loading tokenizer from {args.pretrained_tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_tokenizer)
    logger.info(f"Tokenizer loaded with vocab size: {len(tokenizer)}")
    logger.info(
        f"Special tokens - BOS: {tokenizer.bos_token_id}, EOS: {tokenizer.eos_token_id}"
    )

    # Create or load model
    if resume_info:
        # Load model from checkpoint
        logger.info(f"Loading model from checkpoint: {resume_info['checkpoint_dir']}")
        model = GPT2LMHeadModel.from_pretrained(resume_info["checkpoint_dir"])

        # Count parameters (for logging)
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    else:
        # Create new model
        size_config = args.gpt2_size.split("_")
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=args.seq_length,
            n_ctx=args.seq_length,
            n_embd=int(size_config[2]),
            n_layer=int(size_config[0]),
            n_head=int(size_config[1]),
            use_cache=False,  # Important: disable cache for training
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        model, num_trainable, num_frozen = initialize_model(
            gpt2_config=config,
            pretrained_model_path=args.pretrained_model,
            weights_to_transfer=args.weights_to_transfer,
            weights_to_train=args.weights_to_be_trained,
            device=accelerator.device,
        )

    # Enable gradient checkpointing if specified
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Print model info
    if accelerator.is_main_process:
        logger.info(f"Trainable parameters: {num_trainable}")
        logger.info(f"Frozen parameters: {num_frozen}")
        if not resume_info:
            logger.info(f"Weights transferred: {args.weights_to_transfer}")
            logger.info(f"Weights to be trained: {args.weights_to_be_trained}")

    # Create dataloaders
    train_dataloader, eval_dataloader = create_dataloaders(args, tokenizer)

    # Setup optimizer and scheduler
    optimizer = AdamW(get_grouped_params(model, args), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Resume optimizer and scheduler states if checkpoint exists
    completed_steps = 0
    if resume_info:
        completed_steps = resume_info["completed_steps"]
        optimizer.load_state_dict(resume_info["optimizer_state_dict"])
        lr_scheduler.load_state_dict(resume_info["lr_scheduler_state_dict"])

        # Restore random states for reproducibility
        if resume_info["random_state"] is not None:
            torch.set_rng_state(resume_info["random_state"])
        if torch.cuda.is_available() and resume_info["cuda_random_state"] is not None:
            torch.cuda.set_rng_state(resume_info["cuda_random_state"])

        # Skip already completed steps in dataloader
        steps_to_skip = completed_steps * args.gradient_accumulation_steps
        logger.info(f"Skipping {steps_to_skip} steps in training dataloader")
        train_iter = iter(train_dataloader)
        for _ in range(steps_to_skip):
            next(train_iter)

        logger.info(f"Resuming training from step {completed_steps}")
    else:
        train_iter = iter(train_dataloader)

    # Training loop
    logger.info("Starting training...")
    model.train()
    train_losses = []

    with tqdm(
        total=args.max_train_steps,
        initial=completed_steps,
        desc="Training",
        disable=not accelerator.is_local_main_process,
    ) as pbar:
        for step, batch in enumerate(train_iter, start=1):
            # Forward pass
            outputs = model(batch, labels=batch, use_cache=False)
            loss = outputs.loss / args.gradient_accumulation_steps

            # Backward pass
            accelerator.backward(loss)

            # Track loss
            train_losses.append(loss.detach())

            if step % args.gradient_accumulation_steps == 0:
                # Gradient clipping
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
                pbar.update(1)

                # Log training metrics
                if completed_steps % args.logging_steps == 0:
                    # Use available losses if fewer than logging_steps have accumulated
                    num_losses = min(len(train_losses), args.logging_steps)
                    avg_loss = torch.mean(torch.stack(train_losses[-num_losses:]))
                    metrics = {
                        "loss": avg_loss.item(),
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "perplexity": torch.exp(avg_loss).item(),
                    }
                    log_metrics(accelerator, completed_steps, metrics, "train")

                # Evaluation and checkpointing
                if completed_steps % args.save_checkpoint_steps == 0:
                    logger.info(f"Step {completed_steps}: Running evaluation...")
                    eval_loss, eval_perplexity = evaluate(
                        args, model, eval_dataloader, accelerator
                    )

                    metrics = {"loss": eval_loss, "perplexity": eval_perplexity}
                    log_metrics(accelerator, completed_steps, metrics, "eval")

                    # Save latest checkpoint
                    save_checkpoint(
                        accelerator,
                        model,
                        tokenizer,
                        optimizer,
                        lr_scheduler,
                        completed_steps,
                        metrics,
                        args,
                        checkpoint_type="latest",
                    )

                    # Save best checkpoint if this is the best so far
                    if eval_perplexity < args.best_eval_perplexity:
                        args.best_eval_perplexity = eval_perplexity
                        logger.info(f"New best perplexity: {eval_perplexity:.2f}")
                        save_checkpoint(
                            accelerator,
                            model,
                            tokenizer,
                            optimizer,
                            lr_scheduler,
                            completed_steps,
                            metrics,
                            args,
                            checkpoint_type="best",
                        )

                    if (
                        args.early_stopping_perplexity is not None
                        and eval_perplexity < args.early_stopping_perplexity
                    ):
                        logger.info(
                            f"Early stopping at step {completed_steps} with eval perplexity {eval_perplexity}"
                        )
                        break

                    model.train()

            # Check if training is complete
            if completed_steps >= args.max_train_steps:
                break

    # Final evaluation and save
    logger.info("Training completed. Running final evaluation...")
    eval_loss, eval_perplexity = evaluate(args, model, eval_dataloader, accelerator)

    metrics = {"loss": eval_loss, "perplexity": eval_perplexity}
    log_metrics(accelerator, completed_steps, metrics, "eval")

    # Save final checkpoint as latest
    save_checkpoint(
        accelerator,
        model,
        tokenizer,
        optimizer,
        lr_scheduler,
        completed_steps,
        metrics,
        args,
        checkpoint_type="latest",
    )

    # Also save as best if it's the best
    if eval_perplexity < args.best_eval_perplexity:
        args.best_eval_perplexity = eval_perplexity
        save_checkpoint(
            accelerator,
            model,
            tokenizer,
            optimizer,
            lr_scheduler,
            completed_steps,
            metrics,
            args,
            checkpoint_type="best",
        )

    # Save final model in the main output directory as well
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Final model saved to {args.output_dir}")
        logger.info(f"Best perplexity: {args.best_eval_perplexity:.2f}")

    if args.wandb_enable and accelerator.is_main_process:
        wandb.finish()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments with optional YAML config file support.

    Config file values are used as defaults, CLI arguments override them.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a GPT-2 model on CodeParrot using Accelerate"
    )

    # Config file argument
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. CLI arguments override config file values.",
    )

    # Model configuration
    parser.add_argument(
        "--gpt2_size",
        type=str,
        default="12_12_768",
        choices=["12_12_768", "24_16_1024", "16_8_2048"],
        help="Size of the GPT-2 model (layers_heads_hidden)",
    )
    parser.add_argument(
        "--pretrained_tokenizer",
        type=str,
        default="codeparrot/codeparrot",
        help="Type of pretrained tokenizer to use. Options: 'codeparrot' or a file path.",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="",
        help="Dir to the pretrained model.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--weights_to_transfer",
        default=["attn", "ffn", "ln"],
        nargs="+",
        choices=["attn", "ffn", "embed", "heads", "everything", "ln"],
        help="Which weights to transfer from pretrained model",
    )
    parser.add_argument(
        "--weights_to_be_trained",
        default=["everything"],
        nargs="+",
        choices=["attn", "ffn", "embed", "heads", "everything"],
        help="Which weights to train during fine-tuning",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_name_train",
        type=str,
        default="codeparrot/codeparrot-clean-train",
        help="Name of the training dataset on HuggingFace Hub",
    )
    parser.add_argument(
        "--dataset_name_valid",
        type=str,
        default="codeparrot/codeparrot-clean-valid",
        help="Name of the validation dataset on HuggingFace Hub",
    )
    parser.add_argument(
        "--seq_length", type=int, default=1024, help="Length of each sequence"
    )
    parser.add_argument(
        "--shuffle_buffer",
        type=int,
        default=1000,
        help="Buffer size for shuffling streaming dataset",
    )

    # Training parameters
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size per device for training",
    )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=16,
        help="Batch size per device for validation",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="The scheduler type to use",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=50000,
        help="Total number of training steps to perform",
    )
    parser.add_argument(
        "--max_eval_steps",
        type=int,
        default=100,
        help="Maximum number of evaluation steps to run. Set to -1 for full evaluation",
    )
    parser.add_argument(
        "--early_stopping_perplexity",
        type=float,
        default=None,
        help="Early stopping if eval perplexity is around this value",
    )

    # System parameters
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training mode",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log training metrics every N steps",
    )
    parser.add_argument(
        "--save_checkpoint_steps",
        type=int,
        default=5000,
        help="Save checkpoint every N steps",
    )

    # Weights & Biases
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="codeparrot_pretraining",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_name", type=str, default=None, help="Weights & Biases run name"
    )
    parser.add_argument(
        "--wandb_enable", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument("--note", type=str, default="", help="Note for wandb naming")

    # First parse to get config file path
    args, remaining = parser.parse_known_args()

    # If config file is provided, load it and set as defaults
    if args.config is not None:
        config = load_config(args.config)
        parser.set_defaults(**config)

    # Re-parse with config defaults
    args = parser.parse_args()

    args.wandb_name = args.note

    return args


if __name__ == "__main__":
    main()
