import os
import shutil
from typing import Optional, Tuple

import datasets
from transformers import GPT2Tokenizer


class C4DataModule:
    """Encapsulates C4 dataset loading and preprocessing for GPT-2 training."""

    def __init__(
        self,
        model_name: str = "gpt2",
        eval_subset_size: int = 1000,
        use_c4_1m: bool = False,
        eval_dataset_local_path: Optional[str] = None,
        download_eval_dataset: bool = False,
        overwrite_eval_cache: bool = False,
    ) -> None:
        """
        Initialize the C4 data module.

        Args:
            model_name: GPT-2 model variant for selecting dataset size.
            eval_subset_size: Number of samples to use for evaluation.
            use_c4_1m: Whether to use the 1M-sample C4 dataset.
            eval_dataset_local_path: Local path for cached evaluation dataset.
            download_eval_dataset: Whether to download and cache the eval dataset.
            overwrite_eval_cache: Whether to overwrite existing eval cache.
        """
        self.model_name = model_name
        self.eval_subset_size = eval_subset_size
        self.use_c4_1m = use_c4_1m
        if eval_dataset_local_path is not None:
            expanded = os.path.expanduser(eval_dataset_local_path)
            self.eval_dataset_local_path = os.path.abspath(expanded)
        else:
            self.eval_dataset_local_path = None
        self.download_eval_dataset = download_eval_dataset
        self.overwrite_eval_cache = overwrite_eval_cache

        self.c4_large_path = "./downstream/semantic/data/datasets/c4_gpt2_clean_large"
        self.c4_small_path = "./downstream/semantic/data/datasets/c4_gpt2_clean"
        self.c4_1m_path = "./downstream/semantic/data/datasets/c4_gpt2_clean_large"

        self.c4_dataset_path = self._select_c4_dataset_path()

    def load_train_dataset(self) -> datasets.Dataset:
        """Load and filter the C4 training dataset."""
        dataset = datasets.load_from_disk(self.c4_dataset_path)

        if isinstance(dataset, datasets.DatasetDict):
            if "train" in dataset:
                dataset = dataset["train"]

        print(f"Dataset size before filtering: {len(dataset)}")
        dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0)
        print(f"Dataset size after filtering empty sequences: {len(dataset)}")
        return dataset

    def load_eval_dataset(self) -> Optional[datasets.Dataset]:
        """Load the C4 validation dataset for evaluation."""
        cached_eval_dataset = self._load_cached_eval_dataset()
        if cached_eval_dataset is not None:
            cached_eval_dataset = self._filter_eval_dataset(cached_eval_dataset)
            print(f"Evaluation dataset size: {len(cached_eval_dataset)}")
            return cached_eval_dataset

        eval_dataset = self._load_c4_eval_dataset()

        if eval_dataset is None:
            print("No evaluation dataset loaded - evaluation will be skipped")
            return None

        eval_dataset = self._filter_eval_dataset(eval_dataset)
        self._cache_eval_dataset_if_requested(eval_dataset)
        print(f"Evaluation dataset size: {len(eval_dataset)}")
        return eval_dataset

    def get_position_config(self) -> Tuple[int, str]:
        """Return position embedding config for C4 (always 2048)."""
        return 2048, "Using n_positions=2048 for C4 dataset"

    def _is_large_model(self) -> bool:
        """Check if the model is a large variant (gpt2-large or gpt2-xl)."""
        return "gpt2-xl" in self.model_name or "gpt2-large" in self.model_name

    def _select_c4_dataset_path(self) -> str:
        """Select the appropriate C4 dataset path based on model size."""
        if self.use_c4_1m:
            if os.path.exists(self.c4_1m_path):
                print(f"Using 1M-sample C4 dataset for {self.model_name}")
                return self.c4_1m_path
            print(
                "Requested 1M-sample C4 dataset but path does not exist; falling back to cached subset."
            )

        if self._is_large_model():
            print(f"Using large C4 dataset (275k samples) for {self.model_name}")
            return self.c4_large_path

        print(f"Using standard C4 dataset (100k samples) for {self.model_name}")
        return self.c4_small_path

    def _load_cached_eval_dataset(self) -> Optional[datasets.Dataset]:
        if not self.eval_dataset_local_path:
            return None

        if not os.path.exists(self.eval_dataset_local_path):
            print(
                f"eval_dataset_local_path '{self.eval_dataset_local_path}' "
                "does not exist; falling back to remote loading."
            )
            return None

        try:
            print(
                f"Loading evaluation dataset from disk at "
                f"{self.eval_dataset_local_path}"
            )
            return datasets.load_from_disk(self.eval_dataset_local_path)
        except Exception as exc:
            print(
                "Failed to load evaluation dataset from "
                f"{self.eval_dataset_local_path}: {exc}"
            )
            return None

    def _cache_eval_dataset_if_requested(self, eval_dataset: datasets.Dataset) -> None:
        """Cache the evaluation dataset to disk if requested."""
        if not self.download_eval_dataset:
            return

        if not self.eval_dataset_local_path:
            print(
                "download_eval_dataset=True but eval_dataset_local_path was not "
                "provided; skipping save."
            )
            return

        if os.path.exists(self.eval_dataset_local_path):
            if self.overwrite_eval_cache:
                print(
                    f"Overwriting existing evaluation dataset at "
                    f"{self.eval_dataset_local_path}"
                )
                shutil.rmtree(self.eval_dataset_local_path)
            else:
                print(
                    f"Evaluation dataset already exists at "
                    f"{self.eval_dataset_local_path}; skipping save."
                )
                return

        parent_dir = os.path.dirname(self.eval_dataset_local_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        eval_dataset.save_to_disk(self.eval_dataset_local_path)
        print(f"Saved evaluation dataset to {self.eval_dataset_local_path}")

    @staticmethod
    def _filter_eval_dataset(eval_dataset: datasets.Dataset) -> datasets.Dataset:
        return eval_dataset.filter(lambda x: len(x["input_ids"]) > 0)

    def _load_c4_eval_dataset(self) -> datasets.Dataset:
        print("Loading C4 validation split for evaluation...")
        eval_stream = datasets.load_dataset(
            "allenai/c4", name="en", split="validation", streaming=True
        ).take(self.eval_subset_size)
        eval_dataset = datasets.Dataset.from_list(list(eval_stream))

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        def tokenize_eval(examples):
            return tokenizer(
                examples["text"],
                padding=False,
                truncation=True,
                max_length=2048,
                return_overflowing_tokens=True,
                return_length=True,
            )

        return eval_dataset.map(
            tokenize_eval, batched=True, remove_columns=["text", "timestamp", "url"]
        )
