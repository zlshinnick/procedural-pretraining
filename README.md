# Procedural Pretraining

<p align="center">
  <img src="assets/image.png" width="700">
</p>

Official implementation for **"Procedural Pretraining: Warming Up Language Models with Abstract Data"**.

## Overview

Procedural pretraining trains transformer models on synthetic algorithmic tasks before fine-tuning on downstream tasks. This approach improves efficiency and performance on downstream algorithmic and language modeling tasks.

## Repository Structure

```
procedural-pretraining/
├── procedural_pretraining/    # Pretraining on procedural tasks
│   ├── configs/               # Task configurations
│   └── README.md
├── procedural_data/           # Data generators for procedural tasks
├── downstream/
│   ├── algorithmic_tasks/     # Algorithmic reasoning evaluation
│   │   └── README.md
│   └── semantic/              # Language modeling evaluation (C4, CodeParrot, Math)
│       └── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Procedural Pretraining

Train a model on a procedural task:

```bash
python -m procedural_pretraining.cli --config procedural_pretraining/configs/set.yaml
```

### 2. Downstream Evaluation

Transfer pretrained weights to downstream tasks:

```bash
# Semantic: C4 language modeling
python downstream/semantic/c4.py \
    --config downstream/semantic/configs/c4.yaml \
    --pretrained_path pretrained_models/procedural/set/checkpoint-2500

# Algorithmic: reasoning tasks
python downstream/algorithmic_tasks/experiment_stream.py \
    downstream/algorithmic_tasks/configs/procedural.yaml \
    --pretrained_model_path=pretrained_models/procedural/set/checkpoint-2500
```

## Documentation

- [Procedural Pretraining](procedural_pretraining/README.md) - Pretraining on algorithmic tasks
- [Semantic Downstream Tasks](downstream/semantic/README.md) - C4, CodeParrot, DeepMind Math
- [Algorithmic Downstream Tasks](downstream/algorithmic_tasks/README.md) - Reasoning evaluation

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{jiang2025proceduralpretraining,
  title={Procedural Pretraining: Warming Up Language Models with Abstract Data},
  author={Jiang, Liangze and Shinnick, Zachary and van den Hengel, Anton and Saratchandran, Hemanth and Teney, Damien},
  year={2026},
}

```