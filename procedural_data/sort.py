"""Sort task dataset for sequence manipulation training.

This module provides a procedurally generated dataset where the model must
predict the sorted version of an input sequence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# Number of batches to use for validation dataset.
_VAL_BATCHES = 10


class SortDataset(Dataset):
    """Procedurally generated dataset for the sorting task.

    Each sample is a random sequence of tokens, and the target is the sorted
    version of that sequence.

    Attributes:
        seq_len: Length of each input sequence.
        vocab_size: Number of unique tokens in the vocabulary.
        num_samples: Total number of samples in the dataset.
        seed: Base seed for reproducible generation.
        separator_token: Token ID used as separator (equals vocab_size).
        pad_token: Token ID used for padding (equals vocab_size + 1).
    """

    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        num_samples: int,
        seed: int | None = None,
    ) -> None:
        """Initializes the sort task dataset.

        Args:
            seq_len: Length of each input sequence.
            vocab_size: Number of unique tokens in the vocabulary.
            num_samples: Number of samples in the dataset.
            seed: Base seed for reproducible generation.
        """
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.seed = seed if seed is not None else 0
        self.separator_token = vocab_size
        self.pad_token = vocab_size + 1

    def _generate_sort_sequence(
        self,
        seed: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generates a sequence and its sorted version.

        Args:
            seed: Integer seed to initialize the random state for this sequence.

        Returns:
            A tuple containing:
                - Input sequence tensor.
                - Target sequence tensor.
                - Loss mask tensor.
        """
        local_rng = np.random.RandomState(seed)
        sequence = local_rng.randint(1, self.vocab_size, size=self.seq_len).tolist()
        sorted_sequence = sorted(sequence)

        sequence = sequence + [self.separator_token] + sorted_sequence
        target = sequence[1:] + [self.pad_token]

        loss_mask = torch.zeros(len(sequence), dtype=torch.float)
        sep_idx = (
            sequence.index(self.separator_token)
            if self.separator_token in sequence
            else -1
        )
        if sep_idx >= 0:
            for i in range(sep_idx, sep_idx + self.seq_len + 1):
                if i < len(sequence) and target[i] != self.pad_token:
                    loss_mask[i] = 1.0

        return (
            torch.tensor(sequence, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            loss_mask,
        )

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Retrieves a single sample for the sort task.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A dictionary containing:
                - input_ids: The input sequence tensor.
                - attention_mask: Attention mask (1 for non-pad tokens).
                - targets: Target sequence tensor.
                - loss_mask: Mask indicating valid loss positions.
        """
        sequence, target, loss_mask = self._generate_sort_sequence(self.seed + idx)

        return {
            "input_ids": sequence,
            "attention_mask": (sequence != self.pad_token).long(),
            "targets": target,
            "loss_mask": loss_mask,
        }


def build_dataloader(
    args: Any,
    epoch_seed: int,
    seq_length: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Builds train and validation dataloaders for the sort task dataset.

    Args:
        args: Configuration object with attributes: seed, vocab_size,
            batch_size, steps_per_epoch, num_workers.
        epoch_seed: Seed offset for epoch-specific randomization (e.g., epoch number).
        seq_length: Length of each input sequence.

    Returns:
        A tuple of (train_dataloader, validation_dataloader).
    """
    seed = args.seed + epoch_seed

    train_dataset = SortDataset(
        seq_len=seq_length,
        vocab_size=args.vocab_size,
        num_samples=args.batch_size * args.steps_per_epoch,
        seed=seed,
    )

    val_dataset = SortDataset(
        seq_len=seq_length,
        vocab_size=args.vocab_size,
        num_samples=args.batch_size * _VAL_BATCHES,
        seed=seed + 10000,
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
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
