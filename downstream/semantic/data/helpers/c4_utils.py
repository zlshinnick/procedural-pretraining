import os
import fire
from datasets import load_dataset, Dataset
from transformers import GPT2Tokenizer


def cache_data(
    dataset_name: str,
    out_dir: str,
    tokenizer_name: str = "gpt2",
    c4_samples: int = 100000,
):
    """
    Cache C4 dataset tokenized with GPT-2 tokenizer.

    Args:
        dataset_name: Dataset name (should be 'allenai/c4' or 'c4')
        out_dir: Output directory for the tokenized dataset
        tokenizer_name: Must be 'gpt2' (kept for CLI compatibility)
        c4_samples: Number of C4 samples to process
    """
    if tokenizer_name != "gpt2":
        raise ValueError(f"Only 'gpt2' tokenizer is supported, got '{tokenizer_name}'")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # GPT-2 uses eos_token as pad_token by convention
    tokenizer.pad_token = tokenizer.eos_token

    if dataset_name not in ["c4", "allenai/c4"]:
        raise ValueError(f"Only C4 dataset is supported, got '{dataset_name}'")

    print(f"Loading {c4_samples} samples from C4...")
    dataset = load_dataset("allenai/c4", name="en", split="train", streaming=True)
    dataset = dataset.take(c4_samples)
    dataset = Dataset.from_list(list(dataset))

    # Stack/concatenate C4 examples to create exactly 2048-token sequences
    # This eliminates all padding waste for maximum efficiency
    print("Tokenizing and creating 2048-token sequences...")
    all_tokens = []
    for example in dataset:
        tokens = tokenizer(example["text"], truncation=False, add_special_tokens=False)[
            "input_ids"
        ]
        all_tokens.extend(tokens)

    # Create fixed-length sequences of exactly 2048 tokens
    sequences = []
    for i in range(0, len(all_tokens) - 2048 + 1, 2048):
        sequence = all_tokens[i : i + 2048]
        if len(sequence) == 2048:
            sequences.append(
                {
                    "input_ids": sequence,
                    "attention_mask": [1] * 2048,
                }
            )

    dataset = Dataset.from_list(sequences)
    print(
        f"Created {len(sequences)} sequences of exactly 2048 tokens from C4 (100% content efficiency)"
    )

    os.makedirs(out_dir, exist_ok=True)
    dataset.save_to_disk(out_dir)
    print(f"Saved tokenized dataset to {out_dir}")


if __name__ == "__main__":
    fire.Fire({"cache_data": cache_data})
