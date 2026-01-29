"""
Training script for GPT-2 models on C4 with attention weight transfer support.

Usage examples:
    # Using a config file:
    python downstream/semantic/c4.py --config downstream/semantic/configs/c4_baseline.yaml

    # Using command-line arguments (overrides config file values):
    python downstream/semantic/c4.py --config configs/c4.yaml --max_steps 5000

    # Using only command-line arguments:
    python downstream/semantic/c4.py --model_name gpt2 --output_dir output

"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2Tokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from downstream.semantic.data.c4_dataset import C4DataModule  # noqa: E402
from downstream.utils import initialize_model  # noqa: E402
from procedural_pretraining.utils import SIZE_TO_CONFIG  # noqa: E402

logger = logging.getLogger(__name__)


class SaveAtStepsCallback(TrainerCallback):
    """Custom callback to save model at specific training steps."""

    def __init__(self, save_steps: List[int], output_dir: str):
        """
        Args:
            save_steps: List of steps at which to save the model
            output_dir: Base directory for saving checkpoints
        """
        self.save_steps = sorted(save_steps)
        self.output_dir = output_dir

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each step."""
        if state.global_step in self.save_steps:
            checkpoint_dir = f"{self.output_dir}/checkpoint-{state.global_step}"
            kwargs["model"].save_pretrained(checkpoint_dir)

            if "tokenizer" in kwargs:
                kwargs["tokenizer"].save_pretrained(checkpoint_dir)

            logger.info(f"Saved model at step {state.global_step}")


def _build_gpt2_config(
    base_config: GPT2Config, gpt2_size: Optional[str] = None
) -> GPT2Config:
    """
    Construct a GPT-2 config with n_positions=2048 for C4.
    Optionally override architecture via gpt2_size.
    """
    if gpt2_size is not None:
        if gpt2_size not in SIZE_TO_CONFIG:
            available = ", ".join(sorted(SIZE_TO_CONFIG))
            raise ValueError(
                f"Unknown gpt2_size '{gpt2_size}'. Available options: {available}"
            )
        size_config = SIZE_TO_CONFIG[gpt2_size]
        logger.info(
            "Using custom GPT-2 size %s (layers=%d, heads=%d, hidden=%d)",
            gpt2_size,
            size_config["n_layer"],
            size_config["n_head"],
            size_config["n_embd"],
        )
    else:
        size_config = {
            "n_embd": base_config.n_embd,
            "n_layer": base_config.n_layer,
            "n_head": base_config.n_head,
        }

    return GPT2Config(
        vocab_size=base_config.vocab_size,
        n_embd=size_config["n_embd"],
        n_layer=size_config["n_layer"],
        n_head=size_config["n_head"],
        n_positions=2048,  # C4 uses 2048 token sequences
        resid_pdrop=getattr(base_config, "resid_pdrop", 0.1),
        embd_pdrop=getattr(base_config, "embd_pdrop", 0.1),
        attn_pdrop=getattr(base_config, "attn_pdrop", 0.1),
    )


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
        description="Train a GPT-2 model on C4 with weight transfer support"
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
        "--model_name",
        type=str,
        default="gpt2",
        help="Base GPT-2 model variant (gpt2, gpt2-medium, gpt2-large, gpt2-xl)",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to pretrained model for weight transfer",
    )
    parser.add_argument(
        "--reinit",
        type=bool,
        default=True,
        help="If True, initialize a new model; if False, load pretrained weights",
    )
    parser.add_argument(
        "--gpt2_size",
        type=str,
        default=None,
        help="Optional custom architecture key from SIZE_TO_CONFIG",
    )
    parser.add_argument(
        "--weights_to_transfer",
        nargs="+",
        default=["everything"],
        help="Which weights to transfer (e.g., attn ffn)",
    )
    parser.add_argument(
        "--weights_to_train",
        nargs="+",
        default=["everything"],
        help="Which weights to train",
    )
    parser.add_argument(
        "--initialization_strategy",
        type=str,
        default="retain",
        help="Embedding initialization strategy (retain, average)",
    )

    # Training parameters
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--bsz",
        type=int,
        default=4,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log metrics every N steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for checkpoints and outputs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="Logging destination (wandb, none, etc.)",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="c4_pretraining",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--min_lr_rate",
        type=float,
        default=0.1,
        help="Minimum learning rate ratio for cosine scheduler",
    )
    parser.add_argument(
        "--use_callback",
        action="store_true",
        help="Whether to use custom checkpoint callback",
    )
    parser.add_argument(
        "--use_c4_1m",
        action="store_true",
        help="Use 1M-sample C4 dataset for large models",
    )

    # Eval dataset caching options
    parser.add_argument(
        "--eval_dataset_local_path",
        type=str,
        default=None,
        help="Local path for cached evaluation dataset",
    )
    parser.add_argument(
        "--download_eval_dataset",
        action="store_true",
        help="Whether to download and cache eval dataset",
    )
    parser.add_argument(
        "--overwrite_eval_cache",
        action="store_true",
        help="Whether to overwrite existing eval cache",
    )

    # First parse to get config file path
    args, remaining = parser.parse_known_args()

    # If config file is provided, load it and set as defaults
    if args.config is not None:
        config = load_config(args.config)
        parser.set_defaults(**config)

    # Re-parse with config defaults
    args = parser.parse_args()

    return args


def main() -> None:
    """Main training function."""
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args()
    logger.info("Arguments: %s", vars(args))
    set_seed(args.seed)

    # Configure wandb
    if args.wandb_mode == "offline":
        os.environ.setdefault("WANDB_MODE", "offline")
        logger.info("Enabled W&B offline mode (WANDB_MODE=offline)")
    elif args.wandb_mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
        logger.info("Disabled W&B logging (WANDB_DISABLED=true)")
        args.report_to = None

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_name:
        os.environ["WANDB_NAME"] = args.wandb_name

    callback = SaveAtStepsCallback(
        save_steps=list(range(0, 4000, 100)) + list(range(4000, 10000, 1000)),
        output_dir=args.output_dir,
    )

    data_module = C4DataModule(
        model_name=args.model_name,
        eval_subset_size=1000,
        use_c4_1m=args.use_c4_1m,
        eval_dataset_local_path=args.eval_dataset_local_path,
        download_eval_dataset=args.download_eval_dataset,
        overwrite_eval_cache=args.overwrite_eval_cache,
    )
    train_dataset = data_module.load_train_dataset()
    eval_dataset = data_module.load_eval_dataset()
    logger.info("Training dataset size: %d", len(train_dataset))

    if args.reinit:
        logger.info("Initializing model using initialize_model function...")
        base_config = AutoConfig.from_pretrained(args.model_name)

        n_positions, n_positions_message = data_module.get_position_config()
        logger.info(n_positions_message)

        config = _build_gpt2_config(base_config=base_config, gpt2_size=args.gpt2_size)

        model, num_trainable, num_frozen = initialize_model(
            gpt2_config=config,
            pretrained_model_path=args.pretrained_path,
            weights_to_transfer=args.weights_to_transfer,
            weights_to_train=args.weights_to_train,
            embedding_init_strategy=args.initialization_strategy,
        )

        logger.info("Trainable parameters: %d", num_trainable)
        logger.info("Frozen parameters: %d", num_frozen)
        logger.info("Weights transferred: %s", args.weights_to_transfer)
        logger.info("Weights to be trained: %s", args.weights_to_train)

    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        model = model.cuda()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz if eval_dataset is not None else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        seed=args.seed,
        report_to=args.report_to,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_rate},
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        dataloader_drop_last=True,
        remove_unused_columns=False,
        load_best_model_at_end=True if eval_dataset is not None else False,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
        "processing_class": tokenizer,
    }

    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    trainer = Trainer(**trainer_kwargs)

    if args.use_callback:
        trainer.add_callback(callback)

    trainer.train()


if __name__ == "__main__":
    main()
