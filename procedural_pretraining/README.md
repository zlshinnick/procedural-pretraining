# Procedural Pretraining

Training scripts for pretraining GPT-2 models on procedural data.

## Overview

Procedural pretraining trains transformers on algorithmically generated tasks before fine-tuning on downstream tasks. This module provides the pretraining infrastructure.

### Available Tasks

| Task | Config | Description |
|------|--------|-------------|
| `sort` | `configs/sort.yaml` | Sort sequences of numbers |
| `reverse` | `configs/reverse.yaml` | Reverse input sequences |
| `identity` | `configs/identity.yaml` | Copy input to output |
| `set` | `configs/set.yaml` | Set operations (deduplicate) |
| `union` | `configs/union.yaml` | Union of two sets |
| `delete` | `configs/delete.yaml` | Delete elements from sequence |
| `stack` | `configs/stack.yaml` | Stack operations (push/pop) |
| `dyck` | `configs/dyck.yaml` | k Dyck language (balanced brackets) |
| `dyck_shuffle` | `configs/dyck_shuffle.yaml` | k Shuffle Dyck language |


## Running Pretraining

### Using the CLI

```bash
# Train on sort task
python -m procedural_pretraining.cli --config procedural_pretraining/configs/sort.yaml

# Train on stack task
python -m procedural_pretraining.cli --config procedural_pretraining/configs/stack.yaml

# Train on dyck language
python -m procedural_pretraining.cli --config procedural_pretraining/configs/dyck.yaml
```

### Key Parameters

**Task:**
- `task`: Task name (must match one of the available tasks)
- `seq_length`: Maximum sequence length

**Model Architecture:**
- `gpt2_size`: Architecture spec as `layers_heads_hidden` (e.g., `2_4_16`, `12_12_768`)
- `vocab_size`: Vocabulary size for the task

**Training:**
- `epochs`: Number of training epochs
- `max_steps`: Maximum training steps 
- `steps_per_epoch`: Steps per epoch
- `batch_size`: Training batch size
- `lr`: Learning rate
- `lr_schedule`: Scheduler type (`constant`, `cosine`, `linear`)
- `warmup_steps`: Number of warmup steps
- `weight_decay`: Weight decay coefficient

**Checkpointing:**
- `save_dir`: Directory for saving checkpoints
- `save_every_steps`: Checkpoint frequency
- `save_total_limit`: Maximum checkpoints to keep
- `patience`: Early stopping after N validation checks without improvement

**Logging:**
- `wandb_enable`: Enable W&B logging
- `wandb_project`: W&B project name
- `wandb_name`: W&B run name

**Curriculum Learning:**
- `use_curriculum`: Enable curriculum learning (start with short sequences, increase length)
- `curriculum_start`: Initial sequence length (default: 2)
- `curriculum_step_size`: Sequence length increment when mastery is reached (default: 2)
- `curriculum_mastery_threshold`: Accuracy threshold to advance curriculum (default: 0.99)

**Dyck Language Tasks:**
- `k`: Number of bracket types for Dyck-k language
- `p_open`: Probability of generating an opening bracket
- `max_depth`: Maximum nesting depth (dyck_shuffle only) 


## Data Generation

Data generators can be found at `procedural_data/`:

| Task | Data Module |
|------|-------------|
| sort | `procedural_data/sort.py` |
| reverse | `procedural_data/reverse.py` |
| identity | `procedural_data/identity.py` |
| set | `procedural_data/set.py` |
| union | `procedural_data/union.py` |
| delete | `procedural_data/delete.py` |
| stack | `procedural_data/stack.py` |
| dyck | `procedural_data/dyck.py` |
| dyck_shuffle | `procedural_data/dyck_shuffle.py` |


