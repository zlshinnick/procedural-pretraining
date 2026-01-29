"""
CodeParrot dataset utilities for loading and processing data from HuggingFace.

Uses ConstantLengthDataset for efficient token utilization during pretraining.
"""

import logging
from typing import Iterator

import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that concatenates tokenized text into fixed-length sequences.

    This dataset buffers text samples, tokenizes them, and yields sequences of
    exactly `seq_length` tokens. Samples are concatenated with a separator token
    (BOS token) to maximize token utilization during training.

    Args:
        tokenizer: HuggingFace tokenizer with a `bos_token_id` attribute.
        dataset: Iterable dataset where each item has a "content" key.
        infinite: If True, restart from the beginning when exhausted.
        seq_length: Number of tokens per output sequence.
        num_of_sequences: Number of sequences to buffer before yielding.
        chars_per_token: Estimated characters per token for buffer sizing.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite: bool = False,
        seq_length: int = 1024,
        num_of_sequences: int = 1024,
        chars_per_token: float = 3.6,
    ) -> None:
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.bos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences
        self.epoch = 0
        self.infinite = infinite

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Iterate over the dataset, yielding fixed-length token sequences.

        Yields:
            Tensor of shape (seq_length,) containing token IDs.
        """
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    break
                try:
                    buffer.append(next(iterator)["content"])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        self.epoch += 1
                        logger.info(f"Dataset epoch: {self.epoch}")
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    yield torch.tensor(input_ids)
