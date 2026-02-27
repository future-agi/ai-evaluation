"""Config command for managing CLI configuration."""

import json
import os
from pathlib import Path
from typing import Optional

import typer

from fi.cli.config.defaults import BASIC_TEMPLATE
from fi.cli.utils.console import console, print_error, print_success, print_warning


config_app = typer.Typer(
    name="config",
    help="Manage CLI configuration",
    no_args_is_help=True,
)


@config_app.command("show")
def show_config() -> None:
    """Display current configuration settings."""
    config_items = {
        "FI_API_KEY": _mask_key(os.environ.get("FI_API_KEY")),
        "FI_SECRET_KEY": _mask_key(os.environ.get("FI_SECRET_KEY")),
        "FI_BASE_URL": os.environ.get("FI_BASE_URL", "https://api.futureagi.com"),
        "FI_DEFAULT_MODEL": os.environ.get("FI_DEFAULT_MODEL", "gpt-4o"),
    }

    console.print("[bold]Current Configuration:[/bold]\n")
    for key, value in config_items.items():
        if value:
            console.print(f"  [cyan]{key}[/cyan]: {value}")
        else:
            console.print(f"  [cyan]{key}[/cyan]: [dim]not set[/dim]")


@config_app.command("get")
def get_config(
    key: str = typer.Argument(..., help="Configuration key to get"),
) -> None:
    """Get a specific configuration value."""
    value = os.environ.get(key)
    if value:
        # Mask sensitive keys
        if "KEY" in key.upper() or "SECRET" in key.upper():
            console.print(_mask_key(value))
        else:
            console.print(value)
    else:
        print_warning(f"Configuration key not set: {key}")
        raise typer.Exit(1)


@config_app.command("set")
def set_config(
    key: str = typer.Argument(..., help="Configuration key to set"),
    value: str = typer.Argument(..., help="Value to set"),
    save: bool = typer.Option(
        False,
        "--save", "-s",
        help="Save to .env file in current directory",
    ),
) -> None:
    """
    Set a configuration value.

    Note: Without --save, the value is only set for the current session.
    Use --save to persist the value to a .env file.
    """
    os.environ[key] = value

    if save:
        _save_to_env_file(key, value)
        print_success(f"Saved {key} to .env file")
    else:
        print_success(f"Set {key} for current session")
        console.print("[dim]Use --save to persist to .env file[/dim]")


@config_app.command("init")
def init_config(
    path: Path = typer.Argument(
        Path("."),
        help="Directory to create config file in",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Overwrite existing config file",
    ),
) -> None:
    """Create a default configuration file."""
    config_path = path / "fi-evaluation.yaml"

    if config_path.exists() and not force:
        print_warning(
            f"Configuration file already exists: {config_path}\n"
            "Use --force to overwrite."
        )
        raise typer.Exit(1)

    with open(config_path, "w") as f:
        f.write(BASIC_TEMPLATE)

    print_success(f"Created configuration file: {config_path}")


@config_app.command("env")
def show_env_template() -> None:
    """Show environment variable template."""
    template = """# Future AGI Evaluation Configuration
# Copy this to a .env file and fill in your values

# Required: API credentials
FI_API_KEY=your_api_key_here
FI_SECRET_KEY=your_secret_key_here

# Optional: Custom settings
# FI_BASE_URL=https://api.futureagi.com
# FI_DEFAULT_MODEL=gpt-4o

# Optional: Langfuse integration
# LANGFUSE_SECRET_KEY=your_langfuse_secret
# LANGFUSE_PUBLIC_KEY=your_langfuse_public
# LANGFUSE_HOST=https://cloud.langfuse.com
"""
    console.print(template)


def _mask_key(value: Optional[str]) -> Optional[str]:
    """Mask a sensitive key value."""
    if not value:
        return None
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


def _save_to_env_file(key: str, value: str) -> None:
    """Save a key-value pair to .env file."""
    env_path = Path(".env")

    # Read existing content
    existing_lines = []
    if env_path.exists():
        with open(env_path, "r") as f:
            existing_lines = f.readlines()

    # Update or add the key
    key_found = False
    new_lines = []
    for line in existing_lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}\n")
            key_found = True
        else:
            new_lines.append(line)

    if not key_found:
        new_lines.append(f"{key}={value}\n")

    # Write back
    with open(env_path, "w") as f:
        f.writelines(new_lines)
