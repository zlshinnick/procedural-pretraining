"""Experiment configuration for algorithmictasks.

This module provides configuration management for experiments involving
selective weight transfer between pretrained and randomly initialized
transformer models on algorithmic reasoning tasks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

import yaml


class ExperimentConfig:
    """Configuration class for weight transfer experiments.

    Manages experiment settings including model paths, weight transfer
    specifications, and training parameters. Supports loading from YAML
    files with runtime overrides.

    Attributes:
        COMPONENT_PATTERNS: Mapping from component names to layer name patterns.
        DEFAULTS: Default configuration values.
    """

    COMPONENT_PATTERNS: Dict[str, List[str]] = {
        "attn": ["attn"],
        "ffn": ["mlp"],
        "embed": ["wte", "wpe", "embedding", "lm_head"],
    }

    DEFAULTS: Dict[str, Any] = {
        "pretrained_model_path": None,
        "weights_to_be_transferred": [],
        "weights_to_be_trained": ["everything"],
        "embedding_init_strategy": "average",
        "wandb_project": "algorithmic_tasks",
    }

    def __init__(self, **kwargs: Any) -> None:
        """Initialize configuration with provided parameters.

        Args:
            **kwargs: Configuration parameters to override defaults.

        Raises:
            TypeError: If weight specifications are not lists.
            ValueError: If invalid component names or missing paths.
        """
        self._config: Dict[str, Any] = {**self.DEFAULTS, **kwargs}
        self._validate()

    @classmethod
    def from_yaml(cls, yaml_path: str, **overrides: Any) -> ExperimentConfig:
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.
            **overrides: Parameters to override loaded values.

        Returns:
            ExperimentConfig instance with loaded settings.
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict: Dict[str, Any] = yaml.safe_load(f) or {}
        config_dict.update(overrides)
        return cls(**config_dict)

    def _validate(self) -> None:
        """Validate configuration values.

        Raises:
            TypeError: If weight specifications are not lists.
            ValueError: If component names are invalid or required paths missing.
        """
        weights_to_transfer: Any = self._config["weights_to_be_transferred"]
        weights_to_train: Any = self._config["weights_to_be_trained"]

        if not isinstance(weights_to_transfer, list):
            raise TypeError("weights_to_be_transferred must be a list")
        if not isinstance(weights_to_train, list):
            raise TypeError("weights_to_be_trained must be a list")

        valid_components: Set[str] = set(self.COMPONENT_PATTERNS.keys())
        valid_with_everything: Set[str] = valid_components | {"everything"}

        for component in weights_to_transfer:
            if component not in valid_with_everything:
                raise ValueError(
                    f"Invalid component to transfer: {component}. "
                    f"Valid options: {valid_with_everything}"
                )

        for component in weights_to_train:
            if component not in valid_with_everything:
                raise ValueError(
                    f"Invalid component to train: {component}. "
                    f"Valid options: {valid_with_everything}"
                )

        if weights_to_transfer and not self._config["pretrained_model_path"]:
            raise ValueError(
                "pretrained_model_path must be specified when transferring weights"
            )

    def __getattr__(self, name: str) -> Any:
        """Provide attribute-style access to configuration values.

        Args:
            name: Configuration parameter name.

        Returns:
            Configuration value for the given parameter.

        Raises:
            AttributeError: If parameter does not exist.
        """
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        try:
            return self._config[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'"
            ) from None

    def to_dict(self) -> Dict[str, Any]:
        """Return a copy of the configuration dictionary.

        Returns:
            Dictionary containing all configuration parameters.
        """
        return self._config.copy()
