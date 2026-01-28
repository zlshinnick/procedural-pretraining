"""Stack task dataset for sequence manipulation training.

This module provides a procedurally generated dataset where the model must
predict the final state of a stack after a sequence of push/pop operations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# Number of batches to use for validation dataset.
_VAL_BATCHES = 10


class StackDataset(Dataset):
    """Procedurally generated dataset for the stack task.

    Each sample is a sequence of stack operations (push token or pop=0),
    and the target is the final stack contents.

    Attributes:
        input_seq_length: Length of the input operation sequence.
        seq_length: Total sequence length (2 * input_seq_length).
        vocab_size: Number of unique tokens in the vocabulary.
        num_samples: Total number of samples in the dataset.
        seed: Base seed for reproducible generation.
        separator_token: Token ID used as separator (equals vocab_size).
        pad_token: Token ID used for padding (equals vocab_size + 1).
    """

    def __init__(
        self,
        seq_len: int = 20,
        vocab_size: int = 20,
        num_samples: int = 10000,
        seed: int | None = None,
    ) -> None:
        """Initializes the stack task dataset.

        Args:
            seq_len: Length of the input operation sequence.
            vocab_size: Number of unique tokens in the vocabulary.
            num_samples: Number of samples in the dataset.
            seed: Base seed for reproducible generation.
        """
        self.input_seq_length = seq_len
        self.seq_length = seq_len * 2
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.seed = seed if seed is not None else 0
        self.separator_token = vocab_size
        self.pad_token = vocab_size + 1

    def _generate_stack_sequence(
        self,
        seed: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generates a stack operation sequence and the resulting stack state.

        Args:
            seed: Integer seed to initialize the random state for this sequence.

        Returns:
            A tuple containing:
                - Input sequence tensor.
                - Target sequence tensor.
                - Loss mask tensor.
        """
        local_rng = np.random.RandomState(seed)
        operations = []
        stack = []
        tokens_in_stack = set()

        for step in range(self.input_seq_length):
            if step == self.input_seq_length - 1 and len(stack) > 1:
                token_to_pop = stack.pop()
                tokens_in_stack.remove(token_to_pop)
                operations.append(0)
                continue

            if len(stack) <= 1:
                op = "push"
            else:
                if step < int(self.input_seq_length * 2 / 3):
                    op = "push" if local_rng.random() < 0.75 else "pop"
                else:
                    op = "pop" if local_rng.random() < 0.75 else "push"

            # Check if we have tokens available to push
            available_tokens = list(set(range(1, self.vocab_size)) - tokens_in_stack)
            if op == "push" and not available_tokens:
                # No tokens available to push, must pop instead
                op = "pop"

            if op == "push":
                # By this point, we should have at least one available token
                token_index = local_rng.randint(0, len(available_tokens))
                token = available_tokens[token_index]
                stack.append(token)
                tokens_in_stack.add(token)
                operations.append(token)

            if op == "pop" and stack:  # Only pop if stack is not empty
                token_to_pop = stack.pop()
                tokens_in_stack.remove(token_to_pop)
                operations.append(0)
            elif op == "pop" and not stack:
                # If somehow we need to pop but stack is empty, push instead
                if available_tokens:
                    token_index = local_rng.randint(0, len(available_tokens))
                    token = available_tokens[token_index]
                    stack.append(token)
                    tokens_in_stack.add(token)
                    operations.append(token)

        if len(stack) == 0:
            available_tokens = list(set(range(1, self.vocab_size)) - tokens_in_stack)
            token_index = local_rng.randint(0, len(available_tokens))
            token = available_tokens[token_index]
            stack.append(token)

        # Create the full sequence with separator + stack contents
        sequence = (
            operations + [self.separator_token] + stack[::-1]
        )  # Reverse stack for top-to-bottom order

        # Pad with the pad_token
        if len(sequence) < self.seq_length:
            sequence = sequence + [self.pad_token] * (self.seq_length - len(sequence))
        else:
            sequence = sequence[: self.seq_length]

        # Find the separator position
        sep_idx = -1
        for i, token in enumerate(sequence):
            if token == self.separator_token:
                sep_idx = i
                break

        # Create targets
        targets = []
        for i in range(len(sequence)):
            if i < len(sequence) - 1:
                targets.append(sequence[i + 1])
            else:
                targets.append(self.pad_token)

        # Create a mask where 1 indicates to calculate loss, 0 to ignore
        loss_mask = torch.zeros(self.seq_length, dtype=torch.float)

        # We only want to predict stack tokens (after separator)
        if sep_idx >= 0:
            for i in range(sep_idx, sep_idx + len(stack)):
                if i < self.seq_length and targets[i] != self.pad_token:
                    loss_mask[i] = 1.0

        return (
            torch.tensor(sequence, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            loss_mask,
        )

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Retrieves a single sample for the stack task.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A dictionary containing:
                - input_ids: The input sequence tensor.
                - attention_mask: Attention mask (1 for non-pad tokens).
                - targets: Target sequence tensor.
                - loss_mask: Mask indicating valid loss positions.
        """
        sequence, targets, loss_mask = self._generate_stack_sequence(self.seed + idx)
        attention_mask = torch.ones_like(sequence)

        return {
            "input_ids": sequence,
            "attention_mask": attention_mask,
            "targets": targets,
            "loss_mask": loss_mask,
        }


def build_dataloader(
    args: Any,
    epoch_seed: int,
    seq_length: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Builds train and validation dataloaders for the stack task dataset.

    Args:
        args: Configuration object with attributes: seed, vocab_size,
            batch_size, steps_per_epoch, num_workers.
        epoch_seed: Seed offset for epoch-specific randomization (e.g., epoch number).
        seq_length: Length of the input operation sequence.

    Returns:
        A tuple of (train_dataloader, validation_dataloader).
    """
    seed = args.seed + epoch_seed

    train_dataset = StackDataset(
        seq_len=seq_length,
        vocab_size=args.vocab_size,
        num_samples=args.batch_size * args.steps_per_epoch,
        seed=seed,
    )

    val_dataset = StackDataset(
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
