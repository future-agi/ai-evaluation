"""Configuration file loading and discovery."""

import os
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import ValidationError

from fi.cli.config.schema import FIEvaluationConfig


# Default config file names to look for
CONFIG_FILE_NAMES = [
    "fi-evaluation.yaml",
    "fi-evaluation.yml",
    ".fi-evaluation.yaml",
    ".fi-evaluation.yml",
]


def find_config_file(start_path: Optional[Path] = None) -> Optional[Path]:
    """
    Find a configuration file by searching current directory and parents.

    Args:
        start_path: Starting directory for search (default: current working directory)

    Returns:
        Path to configuration file if found, None otherwise
    """
    if start_path is None:
        start_path = Path.cwd()

    current = start_path.resolve()

    # Search up the directory tree
    while current != current.parent:
        for config_name in CONFIG_FILE_NAMES:
            config_path = current / config_name
            if config_path.exists():
                return config_path
        current = current.parent

    # Check root as well
    for config_name in CONFIG_FILE_NAMES:
        config_path = current / config_name
        if config_path.exists():
            return config_path

    return None


def load_config(
    config_path: Optional[Union[str, Path]] = None
) -> FIEvaluationConfig:
    """
    Load and validate configuration from a YAML file.

    Args:
        config_path: Path to configuration file. If not provided,
                    will search for config file automatically.

    Returns:
        Validated FIEvaluationConfig object

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config file is invalid YAML
        ValidationError: If config doesn't match schema
    """
    if config_path is None:
        config_path = find_config_file()
        if config_path is None:
            raise FileNotFoundError(
                "No configuration file found. "
                "Create a fi-evaluation.yaml file or specify --config path."
            )
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML
    try:
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")

    if raw_config is None:
        raise ValueError("Configuration file is empty")

    # Validate against schema
    try:
        config = FIEvaluationConfig(**raw_config)
    except ValidationError as e:
        raise ValueError(f"Configuration validation failed: {e}")

    return config


def load_test_data(data_path: Union[str, Path]) -> list:
    """
    Load test data from a JSON, JSONL, or CSV file.

    Args:
        data_path: Path to test data file

    Returns:
        List of test case dictionaries

    Raises:
        FileNotFoundError: If data file not found
        ValueError: If data format is unsupported or invalid
    """
    import json
    import csv

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Test data file not found: {data_path}")

    suffix = data_path.suffix.lower()

    if suffix == ".json":
        with open(data_path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return [data]
            return data

    elif suffix == ".jsonl":
        data = []
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    elif suffix == ".csv":
        data = []
        with open(data_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
        return data

    else:
        raise ValueError(
            f"Unsupported data format: {suffix}. "
            "Supported formats: .json, .jsonl, .csv"
        )
