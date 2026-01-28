"""
Utility functions for downstream tasks.

Provides functionality for:
- Loading checkpoints from various formats (HuggingFace, raw PyTorch)
- Transferring specific component weights (attention, MLPs, embeddings, etc.)
- Setting trainable/frozen parameters for fine-tuning experiments
"""

import logging
import os
from typing import List, Optional, Tuple, Union

import torch
from torch.nn import Module
from transformers import GPT2Config, GPT2LMHeadModel

logger = logging.getLogger(__name__)

COMPONENT_PATTERNS = {
    "attn": ["attn.c_attn", "attn.c_proj"],
    "attn+ln": ["attn.c_attn", "attn.c_proj", "ln_1.weight", "ln_1.bias"],
    "ffn": ["mlp"],
    "embed": ["wte", "wpe", "embedding", "lm_head"],
    "heads": ["attn.c_attn"],
    "ln": [
        "ln_1.weight",
        "ln_1.bias",
        "ln_2.weight",
        "ln_2.bias",
        "ln_f.weight",
        "ln_f.bias",
    ],
}


def load_checkpoint_state_dict(pretrained_path: str, model: Module) -> dict:
    """
    Load state dict from a checkpoint file, handling both HuggingFace format directories
    and regular checkpoint files.

    Args:
        pretrained_path (str): Path to the pretrained model checkpoint
        model (Module): The model to get device information from

    Returns:
        dict: The loaded state dictionary
    """
    logger.info("Loading checkpoint from %s...", pretrained_path)

    if os.path.isdir(pretrained_path):
        checkpoint_file = None
        for filename in ["pytorch_model.bin", "model.safetensors"]:
            candidate_path = os.path.join(pretrained_path, filename)
            if os.path.exists(candidate_path):
                checkpoint_file = candidate_path
                break

        if checkpoint_file is None:
            raise FileNotFoundError(
                f"Could not find pytorch_model.bin or model.safetensors in {pretrained_path}"
            )

        logger.info("Loading HuggingFace format from %s", checkpoint_file)
        if checkpoint_file.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(
                checkpoint_file,
                device=str(model.device) if hasattr(model, "device") else "cpu",
            )
        else:
            state_dict = torch.load(
                checkpoint_file,
                map_location=model.device if hasattr(model, "device") else "cpu",
                weights_only=False,
            )
    else:
        checkpoint = torch.load(
            pretrained_path, map_location=model.device, weights_only=False
        )
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get(
                "model", checkpoint.get("state_dict", checkpoint)
            )
        else:
            state_dict = checkpoint

    return state_dict


def load_pretrained_weights(
    model: Module,
    pretrained_path: str,
    component_types_to_transfer: List[str],
    embedding_init_strategy: str = "average",
) -> Module:
    """
    Load specific component weights from a pretrained model.

    Args:
        model (Module): The model to load weights into
        pretrained_path (str): Path to the pretrained model checkpoint
        component_types_to_transfer (List[str]): List of component types to transfer (e.g., ["attn", "ffn"])
        embedding_init_strategy (str): Strategy for initializing word token embeddings when "embed" is transferred.
            - "average": Initialize all tokens to the average of pretrained embeddings
            - "partial_matching": Retain original pretrained embeddings, initialize new tokens with average
            - "retain": Keep original pretrained embeddings (requires matching vocabulary sizes)

    Returns:
        Module: The model with transferred weights
    """

    logger.info("============ Loading Pre Trained Weights ============")
    state_dict = load_checkpoint_state_dict(pretrained_path, model)

    should_init_wte = (
        "embed" in component_types_to_transfer
        or "everything" in component_types_to_transfer
    )

    if should_init_wte:
        if "transformer.wte.weight" in state_dict:
            embedding_matrix = state_dict["transformer.wte.weight"]
        elif "wte.weight" in state_dict:
            embedding_matrix = state_dict["wte.weight"]
        else:
            raise KeyError(
                f"Could not find 'transformer.wte.weight' or 'wte.weight' in state_dict. Available keys: {state_dict.keys()}"
            )

        if embedding_init_strategy == "average":
            avg_embedding = torch.mean(embedding_matrix, dim=0)
            with torch.no_grad():
                if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
                    for i in range(model.transformer.wte.weight.size(0)):
                        model.transformer.wte.weight[i] = avg_embedding
                if hasattr(model, "lm_head"):
                    for i in range(model.lm_head.weight.size(0)):
                        model.lm_head.weight[i] = avg_embedding
            logger.info(
                "Word token embedding components initialized to average of pretrained embeddings"
            )

        elif embedding_init_strategy == "partial_matching":
            pretrained_vocab_size = embedding_matrix.size(0)
            current_vocab_size = model.transformer.wte.weight.size(0)

            if current_vocab_size < pretrained_vocab_size:
                raise ValueError(
                    f"Current vocabulary size ({current_vocab_size}) is smaller than "
                    f"pretrained vocabulary size ({pretrained_vocab_size}). "
                    f"Partial matching expects current vocab to be extended version of pretrained vocab."
                )

            avg_embedding = torch.mean(embedding_matrix, dim=0)

            num_new_tokens = current_vocab_size - pretrained_vocab_size

            with torch.no_grad():
                if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
                    model.transformer.wte.weight[:pretrained_vocab_size] = (
                        embedding_matrix.clone()
                    )
                    for i in range(pretrained_vocab_size, current_vocab_size):
                        model.transformer.wte.weight[i] = avg_embedding.clone()

                if hasattr(model, "lm_head"):
                    if model.lm_head.weight.size(0) != current_vocab_size:
                        raise ValueError(
                            f"lm_head vocabulary size mismatch: expected {current_vocab_size}, "
                            f"got {model.lm_head.weight.size(0)}"
                        )

                    model.lm_head.weight[:pretrained_vocab_size] = (
                        embedding_matrix.clone()
                    )

                    for i in range(pretrained_vocab_size, current_vocab_size):
                        model.lm_head.weight[i] = avg_embedding.clone()

            logger.info(
                "Partial matching completed: %d original tokens retained, %d new tokens initialized to average",
                pretrained_vocab_size,
                num_new_tokens,
            )

        elif embedding_init_strategy == "retain":
            pretrained_vocab_size = embedding_matrix.size(0)
            model_vocab_size = model.transformer.wte.weight.size(0)

            if pretrained_vocab_size != model_vocab_size:
                raise ValueError(
                    f"Cannot use 'retain' strategy: vocabulary size mismatch. "
                    f"Pretrained model has {pretrained_vocab_size} tokens, "
                    f"target model has {model_vocab_size} tokens."
                )

            with torch.no_grad():
                if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
                    model.transformer.wte.weight.data = embedding_matrix.clone()

                if hasattr(model, "lm_head"):
                    lm_head_vocab_size = model.lm_head.weight.size(0)
                    if pretrained_vocab_size != lm_head_vocab_size:
                        raise ValueError(
                            f"Cannot use 'retain' strategy: lm_head vocabulary size mismatch. "
                            f"Pretrained model has {pretrained_vocab_size} tokens, "
                            f"lm_head has {lm_head_vocab_size} tokens."
                        )
                    model.lm_head.weight.data = embedding_matrix.clone()

            logger.info(
                "Word token embedding components retained from pretrained model (%d tokens)",
                pretrained_vocab_size,
            )

        else:
            raise ValueError(
                f"Unknown embedding_init_strategy: '{embedding_init_strategy}'. "
                f"Supported strategies are: 'average', 'partial_matching', 'retain'"
            )

    should_transfer_wpe = (
        "embed" in component_types_to_transfer
        or "everything" in component_types_to_transfer
    )

    if should_transfer_wpe:
        pretrained_context_length = None
        new_model_context_length = None

        if "transformer.wpe.weight" in state_dict:
            pretrained_context_length = state_dict["transformer.wpe.weight"].size(0)
        elif "wpe.weight" in state_dict:
            pretrained_context_length = state_dict["wpe.weight"].size(0)

        if hasattr(model, "transformer") and hasattr(model.transformer, "wpe"):
            new_model_context_length = model.transformer.wpe.weight.size(0)

        if (
            pretrained_context_length is not None
            and new_model_context_length is not None
        ):
            with torch.no_grad():
                if hasattr(model, "transformer") and hasattr(model.transformer, "wpe"):
                    if pretrained_context_length >= new_model_context_length:
                        logger.info(
                            "Transferring first %d position embeddings from pretrained model",
                            new_model_context_length,
                        )
                        if "transformer.wpe.weight" in state_dict:
                            model.transformer.wpe.weight.data[
                                :new_model_context_length
                            ] = state_dict["transformer.wpe.weight"][
                                :new_model_context_length
                            ]
                        elif "wpe.weight" in state_dict:
                            model.transformer.wpe.weight.data[
                                :new_model_context_length
                            ] = state_dict["wpe.weight"][:new_model_context_length]
                    else:
                        logger.info(
                            "Randomly Initializing the WPE since pretrained context length < new model's context length"
                        )
                else:
                    raise ValueError(
                        "Model lacks required transformer or position embedding attributes"
                    )
        else:
            raise ValueError("Could not determine context lengths")

    if "everything" in component_types_to_transfer:
        logger.info("Transferring all weights from checkpoint (except embeddings)")

        # Remove embedding weights and attention bias from state dict to preserve custom initialization
        modified_state_dict = state_dict.copy()
        ignore_substrings = ["wte.weight", "wpe.weight", "lm_head.weight", ".attn.bias"]

        ignore_keys = [
            k
            for k in modified_state_dict.keys()
            if any(sub in k for sub in ignore_substrings)
        ]
        for key in ignore_keys:
            if ".attn.bias" in key:
                logger.debug("Skipping transfer of %s as it is mask buffer", key)
            else:
                logger.debug(
                    "Skipping transfer of %s to preserve custom embedding initialization",
                    key,
                )
            modified_state_dict.pop(key)

        # Load all weights (except embeddings as handled above)
        load_result = model.load_state_dict(modified_state_dict, strict=False)
        logger.info(
            "Load result - Missing keys: %d, Unexpected keys: %d",
            len(load_result.missing_keys),
            len(load_result.unexpected_keys),
        )
        return model

    if "embed" in component_types_to_transfer:
        component_types_to_transfer = [
            comp for comp in component_types_to_transfer if comp != "embed"
        ]

    new_state_dict = {}

    all_patterns = []
    for component_type in component_types_to_transfer:
        if component_type in COMPONENT_PATTERNS:
            all_patterns.extend(COMPONENT_PATTERNS[component_type])
            logger.info(
                "Will transfer weights matching patterns from %s: %s",
                component_type,
                COMPONENT_PATTERNS[component_type],
            )

    for key, value in state_dict.items():
        if "wte.weight" in key or "wpe.weight" in key or "lm_head.weight" in key:
            continue
        if ".attn.bias" in key:
            logger.debug("Skipping transfer of %s as it is attention mask buffer", key)
            continue
        if any(pattern in key for pattern in all_patterns):
            component_type = next(
                (
                    ct
                    for ct, patterns in COMPONENT_PATTERNS.items()
                    if any(pattern in key for pattern in patterns)
                ),
                "unknown",
            )
            logger.debug(
                "Transferring %s weight: %s with shape %s",
                component_type,
                key,
                value.shape,
            )
            new_state_dict[key] = value

    logger.info(
        "Transferring %d weight tensors out of %d total tensors",
        len(new_state_dict),
        len(state_dict),
    )
    load_result = model.load_state_dict(new_state_dict, strict=False)
    logger.info(
        "Load result - Missing keys: %d, Unexpected keys: %d",
        len(load_result.missing_keys),
        len(load_result.unexpected_keys),
    )

    return model


def set_trainable_components(
    model: Module,
    component_types_to_train: List[str],
) -> Tuple[int, int]:
    """
    Set which components of the model should be trainable.

    Args:
        model (Module): The model to configure
        component_types_to_train (List[str]): List of component types to make trainable (e.g., ["embed"])

    Returns:
        Tuple[int, int]: A tuple containing (trainable parameter count, frozen parameter count)
    """
    logger.info("============ Setting Weights To Be Trained ============")

    if "everything" in component_types_to_train:
        logger.info("Training all parameters by default.")

        def should_train_fn(name: str) -> bool:
            return True
    else:
        weights_to_train = []
        for component_type in component_types_to_train:
            if component_type in COMPONENT_PATTERNS:
                weights_to_train.extend(COMPONENT_PATTERNS[component_type])
                logger.info(
                    "Will train weights matching patterns from %s", component_type
                )

        def should_train_fn(name: str) -> bool:
            return any(pattern in name for pattern in weights_to_train)

    trainable_count = 0
    frozen_count = 0

    for name, param in model.named_parameters():
        should_train = should_train_fn(name)
        param.requires_grad = should_train

        if should_train:
            logger.debug("Trainable: %s", name)
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()

    total_params = sum(p.numel() for p in model.parameters())
    if trainable_count + frozen_count != total_params:
        raise ValueError(
            f"Parameter counting mismatch: {trainable_count} trainable + {frozen_count} frozen != {total_params} total"
        )

    logger.info("Parameters: %d trainable, %d frozen", trainable_count, frozen_count)
    return trainable_count, frozen_count


def initialize_model(
    gpt2_config: GPT2Config,
    pretrained_model_path: Optional[str],
    weights_to_transfer: List[str],
    weights_to_train: List[str],
    embedding_init_strategy: str = "average",
    device: Union[str, torch.device] = "cuda",
) -> Tuple[GPT2LMHeadModel, int, int]:
    """
    Initialize and configure a model with specified transfer and training settings.

    Args:
        gpt2_config (GPT2Config): GPT2Config object with model architecture settings
        pretrained_model_path (Optional[str]): Path to pretrained weights, or None if initializing from scratch
        weights_to_transfer (List[str]): List of component types to transfer (e.g., ["attn", "ffn"])
        weights_to_train (List[str]): List of component types to make trainable (e.g., ["embed"])
        embedding_init_strategy (str): Strategy for embedding initialization ("average", "partial_matching", "retain")
        device (Union[str, torch.device]): Device to place model on

    Returns:
        Tuple[GPT2LMHeadModel, int, int]: A tuple containing:
            - The configured model
            - Count of trainable parameters
            - Count of frozen parameters
    """

    if pretrained_model_path:
        if not os.path.exists(pretrained_model_path):
            raise FileNotFoundError(
                f"Pretrained model path does not exist: {pretrained_model_path}"
            )
        if os.path.isdir(pretrained_model_path):
            for filename in ["pytorch_model.bin", "model.safetensors"]:
                if os.path.exists(os.path.join(pretrained_model_path, filename)):
                    break
            else:
                raise FileNotFoundError(
                    f"Could not find pytorch_model.bin or model.safetensors in {pretrained_model_path}"
                )

    model = GPT2LMHeadModel(gpt2_config).to(device)
    model_size = sum(t.numel() for t in model.parameters())
    logger.info("GPT-2 size: %.1fM parameters", model_size / 1_000_000)

    if pretrained_model_path:
        if weights_to_transfer:
            load_pretrained_weights(
                model,
                pretrained_model_path,
                weights_to_transfer,
                embedding_init_strategy,
            )
        else:
            raise ValueError("No weights specified to transfer from pretrained model")
    else:
        logger.info("Initializing model from scratch with random weights")

    trainable_count, frozen_count = set_trainable_components(model, weights_to_train)

    return model, trainable_count, frozen_count
