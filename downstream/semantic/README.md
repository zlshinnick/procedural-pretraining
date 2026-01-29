# Semantic Downstream Tasks

Training scripts for evaluating procedural pretraining on semantic language modeling tasks.

## Overview

| Task | Script | Dataset | Training Framework |
|------|--------|---------|-------------------|
| C4 | `c4.py` | [allenai/c4](https://huggingface.co/datasets/allenai/c4) | HuggingFace Trainer |
| CodeParrot | `codeparrot.py` | [codeparrot/codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean-train) | Accelerate |
| DeepMind Math | `deepmind_math.py` | [DeepMind Mathematics](https://github.com/google-deepmind/mathematics_dataset) | Accelerate |

## Setup

### Prepare Data

#### C4

```bash
bash downstream/semantic/data/scripts/c4.sh
```

This creates tokenized datasets at:
- `downstream/semantic/data/datasets/c4_gpt2_clean/` (100k samples for gpt2/gpt2-medium)
- `downstream/semantic/data/datasets/c4_gpt2_clean_large/` (1M samples for gpt2-large/xl)

#### CodeParrot

Data is streamed from HuggingFace Hub (no preparation needed).

#### DeepMind Math

```bash
bash downstream/semantic/data/scripts/math.sh
```

This downloads and extracts the dataset to:
- `downstream/semantic/data/deepmind_math/mathematics_dataset-v1.0/`

## Running Training

All scripts support YAML config files. CLI arguments override config values.

### C4

```bash
# Single GPU
python downstream/semantic/c4.py --config downstream/semantic/configs/c4.yaml

# Multi-GPU
torchrun --nproc_per_node 4 downstream/semantic/c4.py \
    --config downstream/semantic/configs/c4.yaml
```

### CodeParrot

```bash
# Single GPU
accelerate launch downstream/semantic/codeparrot.py \
    --config downstream/semantic/configs/codeparrot.yaml

# Multi-GPU
accelerate launch --multi_gpu --num_processes 4 downstream/semantic/codeparrot.py \
    --config downstream/semantic/configs/codeparrot.yaml
```

### DeepMind Math

```bash
# Single GPU
accelerate launch downstream/semantic/deepmind_math.py \
    --config downstream/semantic/configs/deepmind_math.yaml

# Multi-GPU
accelerate launch --multi_gpu downstream/semantic/deepmind_math.py \
    --config downstream/semantic/configs/deepmind_math.yaml
```

## Configuration

Example config files are in `downstream/semantic/configs/`:

| Config |
|--------|
| `c4.yaml` | 
| `codeparrot.yaml` |
| `deepmind_math.yaml` | 

### Key Parameters

**Model:**
- `model_name`: GPT-2 variant (`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`)
- `gpt2_size`: Custom architecture (`12_12_768`, `24_16_1024`, `16_8_2048`)
- `pretrained_path` / `pretrained_model`: Path to model for weight transfer

**Weight Transfer:**
- `weights_to_transfer`: Which weights to copy (`attn`, `ffn`, `ln`, `embed`, `everything`)
- `weights_to_train`: Which weights to train (`everything`, or subset)

