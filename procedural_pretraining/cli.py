#!/usr/bin/env python3
"""Unified command-line interface for procedural pretraining.

This module provides a single entry point for training transformer models
on procedural tasks, dispatching to the appropriate
training script based on the task specified in the configuration file.

Usage:
    python cli.py --config configs/union.yaml
    python cli.py --config configs/dyck_shuffle.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import NoReturn

from .utils import load_config_file, merge_config_with_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Task routing: maps task names to their training module
TASK_TO_MODULE = {
    # Dyck language tasks
    "dyck": "dyck",
    "dyck_shuffle": "dyck",
    # Algorithmic tasks
    "stack": "algorithmic",
    "set": "algorithmic",
    "identity": "algorithmic",
    "reverse": "algorithmic",
    "union": "algorithmic",
    "delete": "algorithmic",
    "sort": "algorithmic",
}


def _exit_with_error(message: str) -> NoReturn:
    """Print error to stderr and exit."""
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    """Dispatch to the appropriate training script based on config."""
    parser = argparse.ArgumentParser(
        description="Unified CLI for procedural pretraining",
        epilog="Tasks: " + ", ".join(sorted(TASK_TO_MODULE.keys())),
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Load config and extract task
    try:
        config = load_config_file(args.config)
    except FileNotFoundError:
        _exit_with_error(f"Config not found: {args.config}")
    except Exception as e:
        _exit_with_error(f"Failed to load config: {e}")

    task = config.get("task")
    if not task:
        _exit_with_error("Config must specify 'task'")
    if task not in TASK_TO_MODULE:
        _exit_with_error(f"Unknown task: {task}")

    module = TASK_TO_MODULE[task]
    logger.info("Task: %s | Module: %s", task, module)

    # Prepare sys.argv for the training script's parser
    sys.argv = [sys.argv[0], "--config", args.config]

    # Dispatch to appropriate training script
    if module == "dyck":
        from .dyck import create_parser, main as train

        train(merge_config_with_args(create_parser()))
    else:
        from .algorithmic import create_parser, main as train

        train(merge_config_with_args(create_parser()))


if __name__ == "__main__":
    main()
