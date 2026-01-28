"""Dyck Shuffle Dataset for next token prediction training.

This module provides utilities for generating k-Dyck Shuffle sequences and a
PyTorch Dataset implementation.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Number of batches to use for validation dataset.
_VAL_BATCHES = 10


def generate_dyck_shuffle(
    k: int,
    length: int,
    p_open: float = 0.5,
    max_depth: int = 16,
) -> list[int]:
    """Generates a k-Dyck Shuffle sequence.

    Args:
        k: Number of different types of brackets.
        length: Length of the sequence.
        p_open: Probability of opening a new bracket.
        max_depth: Maximum nesting depth allowed.

    Returns:
        A list of integers representing the sequence. Tokens 0 to k-1 represent
        opening brackets, and tokens k to 2k-1 represent closing brackets.
    """
    sequence: list[int] = []
    counts: list[int] = [0] * k  # Track open brackets of each type

    while len(sequence) < length:
        depth = sum(counts)

        # Must open if all brackets are closed
        if depth == 0:
            bracket = random.randint(0, k - 1)
            sequence.append(bracket)
            counts[bracket] += 1
            continue

        # If at max depth, force a close
        if depth >= max_depth:
            open_brackets = [i for i, count in enumerate(counts) if count > 0]
            bracket = random.choice(open_brackets)
            sequence.append(bracket + k)
            counts[bracket] -= 1
            continue

        # Randomly choose to open or close
        if random.random() < p_open and depth < max_depth:
            bracket = random.randint(0, k - 1)
            sequence.append(bracket)
            counts[bracket] += 1
        else:
            # Close an existing bracket
            open_brackets = [i for i, count in enumerate(counts) if count > 0]
            bracket = random.choice(open_brackets)
            sequence.append(bracket + k)
            counts[bracket] -= 1

    return sequence


class DyckShuffleDataset(Dataset):
    """PyTorch Dataset for k-Dyck Shuffle sequences with next token prediction.

    Attributes:
        k: Number of bracket types.
        seq_length: Sequence length.
        p_open: Probability of opening a new bracket.
        max_depth: Maximum nesting depth.
        num_samples: Total number of samples in the dataset.
        pad_token: Token ID used for padding (equals 2*k).
        samples: Pre-generated list of sequences.
    """

    def __init__(
        self,
        k: int = 10,
        seq_length: int = 2048,
        p_open: float = 0.5,
        max_depth: int = 16,
        num_samples: int = 10000,
    ) -> None:
        """Initializes dataset for k-Dyck Shuffle sequences.

        Args:
            k: Number of bracket types.
            seq_length: Maximum sequence length.
            p_open: Probability of opening a bracket.
            max_depth: Maximum nesting depth.
            num_samples: Number of samples in the dataset.
        """
        self.k = k
        self.seq_length = seq_length
        self.p_open = p_open
        self.max_depth = max_depth
        self.num_samples = num_samples
        self.pad_token = 2 * k
        self.samples: list[list[int]] = []
        for _ in range(num_samples):
            sequence = generate_dyck_shuffle(k, seq_length, p_open, max_depth)
            self.samples.append(sequence)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Retrieves a single sample for next token prediction.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A dictionary containing:
                - input_ids: The input sequence tensor.
                - attention_mask: Attention mask (all ones).
                - targets: Target sequence (input shifted by one position).
                - loss_mask: Mask indicating valid loss positions.
        """
        sequence = torch.tensor(self.samples[idx], dtype=torch.long)
        targets = torch.cat([sequence[1:], torch.tensor([self.pad_token])])
        attention_mask = torch.ones_like(sequence)
        loss_mask = torch.ones_like(sequence, dtype=torch.float)
        loss_mask[-1] = 0.0

        return {
            "input_ids": sequence,
            "attention_mask": attention_mask,
            "targets": targets,
            "loss_mask": loss_mask,
        }


def build_dataloader(
    args: Any,
    epoch_seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Builds train and validation dataloaders for the Dyck Shuffle dataset.

    Args:
        args: Configuration object with attributes: seed, k, seq_length,
            p_open, max_depth, batch_size, steps_per_epoch, num_workers.
        epoch_seed: Seed offset for epoch-specific randomization (e.g., epoch number).

    Returns:
        A tuple of (train_dataloader, validation_dataloader).
    """
    seed = args.seed + epoch_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = DyckShuffleDataset(
        k=args.k,
        seq_length=args.seq_length,
        p_open=args.p_open,
        max_depth=args.max_depth,
        num_samples=args.batch_size * args.steps_per_epoch,
    )

    val_dataset = DyckShuffleDataset(
        k=args.k,
        seq_length=args.seq_length,
        p_open=args.p_open,
        max_depth=args.max_depth,
        num_samples=args.batch_size * _VAL_BATCHES,
    )

    logger.info(
        "Dataset created: k=%d, length=%d, seed=%s", args.k, args.seq_length, seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
