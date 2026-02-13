"""Configuration management utilities."""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If configuration file does not exist
        yaml.YAMLError: If configuration file is malformed
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path to save the configuration file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved configuration to {output_path}")

    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override taking precedence.

    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def get_nested_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a nested value from configuration using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the value (e.g., "model.lightgbm.learning_rate")
        default: Default value if key is not found

    Returns:
        Value at the specified path or default

    Example:
        >>> config = {"model": {"lightgbm": {"learning_rate": 0.05}}}
        >>> get_nested_value(config, "model.lightgbm.learning_rate")
        0.05
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def set_nested_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Set a nested value in configuration using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the value (e.g., "model.lightgbm.learning_rate")
        value: Value to set

    Example:
        >>> config = {"model": {"lightgbm": {}}}
        >>> set_nested_value(config, "model.lightgbm.learning_rate", 0.05)
        >>> config["model"]["lightgbm"]["learning_rate"]
        0.05
    """
    keys = key_path.split(".")
    current = config

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value
