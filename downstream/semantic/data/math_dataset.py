import logging
import os
import random
import string

import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)

MODULES = {
    "interpolation": [
        "arithmetic__add_or_sub",
        "arithmetic__add_or_sub_in_base",
        "arithmetic__add_sub_multiple",
        "arithmetic__div",
        "arithmetic__mixed",
        "arithmetic__mul",
        "arithmetic__mul_div_multiple",
        "arithmetic__nearest_integer_root",
        "arithmetic__simplify_surd",
        "algebra__linear_1d",
        "algebra__linear_1d_composed",
        "algebra__linear_2d",
        "algebra__linear_2d_composed",
        "algebra__polynomial_roots",
        "algebra__polynomial_roots_composed",
        "algebra__sequence_next_term",
        "algebra__sequence_nth_term",
        "numbers__base_conversion",
        "numbers__div_remainder",
        "numbers__div_remainder_composed",
        "numbers__gcd",
        "numbers__gcd_composed",
        "numbers__is_factor",
        "numbers__is_factor_composed",
        "numbers__is_prime",
        "numbers__is_prime_composed",
        "numbers__lcm",
        "numbers__lcm_composed",
        "numbers__list_prime_factors",
        "numbers__round_number",
        "polynomials__evaluate",
        "polynomials__collect",
        "calculus__differentiate",
        "probability__swr_p_sequence",
        "calculus__differentiate_composed",
        "numbers__list_prime_factors_composed",
        "comparison__closest",
        "numbers__place_value",
        "numbers__round_number_composed",
        "measurement__conversion",
        "comparison__sort",
        "polynomials__expand",
        "comparison__pair",
        "polynomials__simplify_power",
        "comparison__closest_composed",
        "polynomials__coefficient_named",
        "polynomials__add",
        "comparison__pair_composed",
        "comparison__sort_composed",
        "comparison__kth_biggest",
        "comparison__kth_biggest_composed",
        "measurement__time",
        "polynomials__compose",
        "probability__swr_p_level_set",
        "polynomials__evaluate_composed",
        "numbers__place_value_composed",
    ],
    "extrapolation": [
        "algebra__polynomial_roots_big",
        "arithmetic__div_big",
        "arithmetic__mul_div_multiple_longer",
        "comparison__sort_more",
        "numbers__round_number_big",
        "arithmetic__add_or_sub_big",
        "arithmetic__mixed_longer",
        "comparison__closest_more",
        "measurement__conversion",
        "probability__swr_p_level_set_more_samples",
        "arithmetic__add_sub_multiple_longer",
        "arithmetic__mul_big",
        "comparison__kth_biggest_more",
        "numbers__place_value_big",
        "probability__swr_p_sequence_more_samples",
    ],
}


class CharacterLevelMathTokenizer:
    """Character-level tokenizer for mathematical reasoning tasks"""

    def __init__(self):
        # Define vocabulary as mentioned in the papers
        # 95 characters: upper/lowercase, digits, punctuation
        self.vocab = (
            string.ascii_letters  # a-z, A-Z (52 chars)
            + string.digits  # 0-9 (10 chars)
            + string.punctuation  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ (32 chars)
            + " "  # space (1 char)
        )

        # Create character to index mapping
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.sep_token = "<SEP>"

        # Add special tokens to vocabulary
        self.char_to_idx[self.pad_token] = len(self.vocab)
        self.char_to_idx[self.unk_token] = len(self.vocab) + 1
        self.char_to_idx[self.bos_token] = len(self.vocab) + 2
        self.char_to_idx[self.sep_token] = len(self.vocab) + 3

        self.idx_to_char[len(self.vocab)] = self.pad_token
        self.idx_to_char[len(self.vocab) + 1] = self.unk_token
        self.idx_to_char[len(self.vocab) + 2] = self.bos_token
        self.idx_to_char[len(self.vocab) + 3] = self.sep_token

        self.vocab_size = len(self.char_to_idx)
        self.pad_token_id = self.char_to_idx[self.pad_token]
        self.bos_token_id = self.char_to_idx[self.bos_token]
        self.sep_token_id = self.char_to_idx[self.sep_token]

    def encode(self, text):
        """Convert text to list of character indices"""
        return [
            self.char_to_idx.get(char, self.char_to_idx[self.unk_token])
            for char in text
        ]

    def decode(self, indices):
        """Convert list of indices back to text"""
        return "".join(
            [
                self.idx_to_char.get(int(idx), self.unk_token)
                for idx in indices
                if idx != self.pad_token_id
            ]
        )

    def __len__(self):
        """Return the vocabulary size"""
        return self.vocab_size


class MathDataset(IterableDataset):
    """
    Dataset that uniformly samples from separate problem type files
    """

    def __init__(
        self,
        tokenizer,
        data_dir,
        infinite=True,
        seq_length=512,
        num_of_sequences=1024,
        chars_per_token=1.0,
        allow_partial_sequences=False,
        sampling_weights=None,  # Optional: custom weights for each module
        has_difficulty_dirs=False,  # True for train, False for test
        difficulty_weights=None,  # Optional: weights for ['easy', 'medium', 'hard']
        split=None,
    ):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences
        self.epoch = 0
        self.infinite = infinite
        self.allow_partial_sequences = allow_partial_sequences
        self.has_difficulty_dirs = has_difficulty_dirs
        if split == "interpolation" or split == "train":
            self.modules = MODULES["interpolation"]
        elif split == "extrapolation":
            self.modules = MODULES["extrapolation"]
        else:
            raise ValueError(f"Unknown split: {split}")

        # Set up difficulty levels and weights
        if has_difficulty_dirs:
            self.difficulties = ["easy", "medium", "hard"]
            if difficulty_weights is None:
                self.difficulty_weights = [1.0] * len(self.difficulties)
            else:
                self.difficulty_weights = difficulty_weights
        else:
            self.difficulties = ["no_difficulty"]  # Use string instead of None
            self.difficulty_weights = [1.0]

        # Find all problem type files
        self.problem_files = {}
        if has_difficulty_dirs:
            # Structure: data_dir/train-{difficulty}/{module}.txt
            for difficulty in self.difficulties:
                difficulty_dir = os.path.join(data_dir, f"train-{difficulty}")
                if os.path.exists(difficulty_dir):
                    for filename in os.listdir(difficulty_dir):
                        if filename.endswith(".txt"):
                            module_name = filename.replace(".txt", "")
                            if module_name in self.modules:
                                key = f"{difficulty}_{module_name}"
                                self.problem_files[key] = os.path.join(
                                    difficulty_dir, filename
                                )
        else:
            # Structure: data_dir/{module}.txt
            for filename in os.listdir(data_dir):
                if filename.endswith(".txt"):
                    module_name = filename.replace(".txt", "")
                    if module_name in self.modules:
                        self.problem_files[module_name] = os.path.join(
                            data_dir, filename
                        )

        print(f"Found {len(self.problem_files)} problem type files")

        # Set up module sampling weights
        if sampling_weights is None:
            self.sampling_weights = [1.0] * len(self.modules)
        else:
            self.sampling_weights = sampling_weights

        # Get special token ids
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0)
        self.bos_token_id = getattr(
            tokenizer, "bos_token_id", tokenizer.char_to_idx["<BOS>"]
        )
        self.sep_token_id = getattr(
            tokenizer, "sep_token_id", tokenizer.char_to_idx["<SEP>"]
        )

        # Cache for file iterators
        self.file_iterators = {}

    def _get_file_iterator(self, file_key):
        """Get or create iterator for a specific file"""
        if file_key not in self.file_iterators:
            self.file_iterators[file_key] = self._create_file_iterator(file_key)
        return self.file_iterators[file_key]

    def _create_file_iterator(self, file_key):
        """Create iterator for a specific problem file using line-by-line streaming"""
        filepath = self.problem_files[file_key]

        def file_iterator():
            while True:  # Infinite iterator
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        # Read two lines at a time (question, answer)
                        while True:
                            question = f.readline()
                            if not question:
                                break
                            answer = f.readline()
                            if not answer:
                                break
                            question = question.strip()
                            answer = answer.strip()

                            # Extract module name from file_key
                            if self.has_difficulty_dirs:
                                module_name = file_key.split("_", 1)[
                                    1
                                ]  # Remove difficulty prefix
                            else:
                                module_name = file_key

                            yield {
                                "question": question,
                                "answer": answer,
                                "module": module_name,
                            }
                    if not self.infinite:
                        break
                except Exception as e:
                    logger.error(f"Error reading {filepath}: {e}")
                    break

        return file_iterator()

    def _sample_problem(self):
        """Sample a problem uniformly from all module types and difficulties"""
        if self.has_difficulty_dirs:
            # Two-stage sampling: first difficulty, then module
            difficulty = random.choices(
                self.difficulties, weights=self.difficulty_weights
            )[0]
            module = random.choices(self.modules, weights=self.sampling_weights)[0]

            # Build file key
            file_key = f"{difficulty}_{module}"

            # Check if this combination exists
            if file_key not in self.problem_files:
                # Fallback to any available difficulty for this module
                available_keys = [
                    k for k in self.problem_files.keys() if k.endswith(f"_{module}")
                ]
                if not available_keys:
                    logger.error(f"No files found for module {module}")
                    return None
                file_key = random.choice(available_keys)
                difficulty = file_key.split("_")[0]
        else:
            # Sample module directly
            module = random.choices(self.modules, weights=self.sampling_weights)[0]
            file_key = module
            difficulty = "no_difficulty"
            if file_key not in self.problem_files:
                logger.error(f"File not found for module {module}")
                return None

        # Get iterator for this file
        iterator = self._get_file_iterator(file_key)

        try:
            problem = next(iterator)
            # Add difficulty info if available
            problem["difficulty"] = difficulty
            return problem
        except StopIteration:
            # Recreate iterator if exhausted
            self.file_iterators[file_key] = self._create_file_iterator(file_key)
            problem = next(self.file_iterators[file_key])
            problem["difficulty"] = difficulty
            return problem

    def _format_problem(self, problem):
        """Format a single math problem with special tokens"""
        # Tokenize question and answer separately
        question_tokens = self.tokenizer.encode(problem["question"])
        answer_tokens = self.tokenizer.encode(problem["answer"])

        # Create formatted sequence: <bos><question><sep><answer>
        formatted_tokens = (
            [self.bos_token_id] + question_tokens + [self.sep_token_id] + answer_tokens
        )

        # Create mask: no loss on <bos>, question, and <sep>; loss only on answer
        mask = [0] + [0] * len(question_tokens) + [0] + [1] * len(answer_tokens)

        return formatted_tokens, mask

    def __iter__(self):
        more_examples = True

        while more_examples:
            buffer_char_len = 0
            current_tokens = []
            current_masks = []
            current_modules = []
            current_difficulties = []

            # Fill buffer by sampling uniformly
            while buffer_char_len < self.input_characters:
                try:
                    # Sample a problem uniformly from all types
                    problem = self._sample_problem()
                    if problem is None:  # Skip if sampling failed
                        continue

                    tokens, mask = self._format_problem(problem)

                    # Check if we can fit this problem in current sequence
                    if len(current_tokens) + len(tokens) <= self.seq_length:
                        current_tokens.extend(tokens)
                        current_masks.extend(mask)
                        current_modules.append(problem["module"])
                        current_difficulties.append(problem["difficulty"])
                        buffer_char_len += len(tokens)
                    else:
                        # Yield current sequence if it has content
                        if current_tokens:
                            # Pad to seq_length
                            padding_needed = self.seq_length - len(current_tokens)
                            current_tokens.extend([self.pad_token_id] * padding_needed)
                            current_masks.extend([0] * padding_needed)

                            # Calculate module statistics
                            module_counts = dict.fromkeys(self.modules, 0)
                            for m in current_modules:
                                module_counts[m] += 1
                            # Calculate difficulty statistics
                            difficulty_counts = dict.fromkeys(self.difficulties, 0)
                            for d in current_difficulties:
                                difficulty_counts[d] += 1

                            yield {
                                "input_ids": torch.tensor(
                                    current_tokens, dtype=torch.long
                                ),
                                "loss_mask": torch.tensor(
                                    current_masks, dtype=torch.long
                                ),
                                "labels": torch.tensor(
                                    current_tokens, dtype=torch.long
                                ),
                                "module_stats": torch.tensor(
                                    list(module_counts.values()), dtype=torch.long
                                ),
                                "difficulty_stats": torch.tensor(
                                    list(difficulty_counts.values()), dtype=torch.long
                                ),
                            }

                        # Start new sequence with current problem
                        current_tokens = tokens.copy()
                        current_masks = mask.copy()
                        current_modules = [problem["module"]]
                        current_difficulties = [problem["difficulty"]]
                        buffer_char_len = len(tokens)

                except Exception as e:
                    logger.error(f"Error sampling problem: {e}")
                    if not self.infinite:
                        more_examples = False
                    break

            # Handle remaining tokens
            if current_tokens and (
                self.allow_partial_sequences
                or len(current_tokens) >= self.seq_length // 2
            ):
                padding_needed = self.seq_length - len(current_tokens)
                current_tokens.extend([self.pad_token_id] * padding_needed)
                current_masks.extend([0] * padding_needed)

                # Calculate module statistics
                module_counts = dict.fromkeys(self.modules, 0)
                for m in current_modules:
                    module_counts[m] += 1
                # Calculate difficulty statistics
                difficulty_counts = dict.fromkeys(self.difficulties, 0)
                for d in current_difficulties:
                    difficulty_counts[d] += 1

                yield {
                    "input_ids": torch.tensor(current_tokens, dtype=torch.long),
                    "loss_mask": torch.tensor(current_masks, dtype=torch.long),
                    "labels": torch.tensor(current_tokens, dtype=torch.long),
                    "module_stats": torch.tensor(
                        list(module_counts.values()), dtype=torch.long
                    ),
                    "difficulty_stats": torch.tensor(
                        list(difficulty_counts.values()), dtype=torch.long
                    ),
                }
