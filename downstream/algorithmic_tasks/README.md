# Algorithmic Tasks

Evaluation scripts for testing procedurally pretrained models on algorithmic reasoning tasks.

## Overview

| Task | File | Task ID |
|------|------|---------|
| Needle In a Haystack | `task_condattendstream.py` | condattendstream_30 |
| Addition | `task_forwardaddition.py` | forwardaddition_5 |
| Reversed Addition | `task_additionstream.py` | additionstream_10 |
| Multiplication | `task_multiplication.py` | multiplication_5 |


## Running Experiments

The main training script is `experiment_stream.py`. It uses a configurator system that supports:
1. YAML config files
2. Command-line overrides

### Basic Usage

```bash
# Baseline Example
python downstream/algorithmic_tasks/experiment_stream.py \
    downstream/algorithmic_tasks/configs/baseline.yaml

# Procedural Model Example
python downstream/algorithmic_tasks/experiment_stream.py \
    downstream/algorithmic_tasks/configs/procedural.yaml \
```

## Configuration

### Example Config Files

| Config | Description |
|--------|-------------|
| `configs/baseline.yaml` | Train from scratch (no pretraining) |
| `configs/procedural.yaml` | Transfer weights from pretrained model |

### Key Parameters

**Task Selection:**
- `task`: Task name with difficulty (e.g., `condattendstream_30`, `forwardaddition_5`, `additionstream_10`, `multiplication_5`)

**Weight Transfer:**
- `pretrained_model_path`: Path to pretrained model
- `weights_to_be_transferred`: Which weights to transfer (`everything`, `attn`, `ffn`, `ln`, [`attn`, `ln`, `ffn`].)
- `weights_to_be_trained`: Which weights to train
- `embedding_init_strategy`: How to initialize embeddings (`average`, `retain`)