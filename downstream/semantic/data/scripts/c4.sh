echo "Creating GPT-2 tokenized version of C4 (100k samples for base/medium models)..."
python3 -m downstream.semantic.data.helpers.c4_utils cache_data \
    --dataset_name allenai/c4 \
    --out_dir ./downstream/semantic/data/datasets/c4_gpt2_clean \
    --tokenizer_name gpt2 \
    --c4_samples 100000

echo "Creating GPT-2 tokenized version of C4 (1M samples for large/xl models)..."
python3 -m downstream.semantic.data.helpers.c4_utils cache_data \
    --dataset_name allenai/c4 \
    --out_dir ./downstream/semantic/data/datasets/c4_gpt2_clean_large \
    --tokenizer_name gpt2 \
    --c4_samples 1000000