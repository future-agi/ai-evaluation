"""Export and import functionality for AutoEval configurations.

Supports YAML and JSON formats for version-controlled configurations.
"""

import json
from pathlib import Path
from typing import Union

from .config import AutoEvalConfig


def export_yaml(config: AutoEvalConfig, path: Union[str, Path]) -> None:
    """
    Export configuration to YAML file.

    Args:
        config: AutoEvalConfig to export
        path: Path to write YAML file

    Example:
        export_yaml(config, "eval_config.yaml")
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for YAML export. Install with: pip install pyyaml")

    data = config.to_dict()
    path = Path(path)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def export_json(config: AutoEvalConfig, path: Union[str, Path], indent: int = 2) -> None:
    """
    Export configuration to JSON file.

    Args:
        config: AutoEvalConfig to export
        path: Path to write JSON file
        indent: Indentation level for pretty printing

    Example:
        export_json(config, "eval_config.json")
    """
    data = config.to_dict()
    path = Path(path)

    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def load_yaml(path: Union[str, Path]) -> AutoEvalConfig:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        AutoEvalConfig instance

    Example:
        config = load_yaml("eval_config.yaml")
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for YAML import. Install with: pip install pyyaml")

    path = Path(path)

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return AutoEvalConfig.from_dict(data)


def load_json(path: Union[str, Path]) -> AutoEvalConfig:
    """
    Load configuration from JSON file.

    Args:
        path: Path to JSON configuration file

    Returns:
        AutoEvalConfig instance

    Example:
        config = load_json("eval_config.json")
    """
    path = Path(path)

    with open(path, "r") as f:
        data = json.load(f)

    return AutoEvalConfig.from_dict(data)


def load_config(path: Union[str, Path]) -> AutoEvalConfig:
    """
    Load configuration from file (auto-detect format).

    Determines format from file extension (.yaml, .yml, .json).

    Args:
        path: Path to configuration file

    Returns:
        AutoEvalConfig instance

    Raises:
        ValueError: If file extension not recognized

    Example:
        config = load_config("eval_config.yaml")
        config = load_config("eval_config.json")
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        return load_yaml(path)
    elif suffix == ".json":
        return load_json(path)
    else:
        raise ValueError(
            f"Unknown file format: {suffix}. "
            "Supported formats: .yaml, .yml, .json"
        )


def to_yaml_string(config: AutoEvalConfig) -> str:
    """
    Convert configuration to YAML string.

    Args:
        config: AutoEvalConfig to convert

    Returns:
        YAML string representation

    Example:
        yaml_str = to_yaml_string(config)
        print(yaml_str)
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    data = config.to_dict()
    return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)


def to_json_string(config: AutoEvalConfig, indent: int = 2) -> str:
    """
    Convert configuration to JSON string.

    Args:
        config: AutoEvalConfig to convert
        indent: Indentation level for pretty printing

    Returns:
        JSON string representation

    Example:
        json_str = to_json_string(config)
        print(json_str)
    """
    data = config.to_dict()
    return json.dumps(data, indent=indent)


def from_yaml_string(yaml_str: str) -> AutoEvalConfig:
    """
    Load configuration from YAML string.

    Args:
        yaml_str: YAML string to parse

    Returns:
        AutoEvalConfig instance

    Example:
        config = from_yaml_string(yaml_content)
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    data = yaml.safe_load(yaml_str)
    return AutoEvalConfig.from_dict(data)


def from_json_string(json_str: str) -> AutoEvalConfig:
    """
    Load configuration from JSON string.

    Args:
        json_str: JSON string to parse

    Returns:
        AutoEvalConfig instance

    Example:
        config = from_json_string(json_content)
    """
    data = json.loads(json_str)
    return AutoEvalConfig.from_dict(data)
