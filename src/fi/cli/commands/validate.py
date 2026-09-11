"""Validate command for checking configuration files."""

import os
from pathlib import Path
from typing import Optional

import typer

from fi.cli.config.loader import load_config, load_test_data, find_config_file
from fi.cli.utils.console import console, print_error, print_success, print_warning


def validate(
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file",
    ),
    strict: bool = typer.Option(
        False,
        "--strict", "-s",
        help="Enable strict validation mode",
    ),
) -> None:
    """
    Validate configuration file and test data.

    Checks:
        - YAML syntax validity
        - Template name existence
        - Required input fields for templates
        - Data file accessibility
        - API key presence (warning if missing)
    """
    errors = []
    warnings = []

    # Find config file
    if config:
        config_path = Path(config)
    else:
        config_path = find_config_file()

    if not config_path:
        print_error("No configuration file found.")
        raise typer.Exit(1)

    console.print(f"[dim]Validating: {config_path}[/dim]\n")

    # 1. Load and validate config
    try:
        eval_config = load_config(config_path)
        console.print("[green]✓[/green] Configuration file syntax is valid")
    except FileNotFoundError as e:
        errors.append(f"Configuration file not found: {e}")
    except ValueError as e:
        errors.append(f"Configuration validation failed: {e}")

    if errors:
        _print_validation_results(errors, warnings, strict)
        raise typer.Exit(1)

    # 2. Validate templates exist
    from fi.evals import templates as templates_module
    from fi.evals.templates import EvalTemplate

    available_templates = set()
    for name in dir(templates_module):
        obj = getattr(templates_module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, EvalTemplate)
            and obj is not EvalTemplate
            and hasattr(obj, "eval_name")
        ):
            available_templates.add(obj.eval_name)

    for eval_def in eval_config.evaluations:
        templates = eval_def.templates or ([eval_def.template] if eval_def.template else [])
        for template in templates:
            if template and template not in available_templates:
                errors.append(
                    f"Unknown template '{template}' in evaluation '{eval_def.name}'. "
                    f"Run 'fi list templates' to see available templates."
                )

    if not errors:
        console.print("[green]✓[/green] All template names are valid")

    # 3. Validate data files
    data_files_valid = True
    for eval_def in eval_config.evaluations:
        data_path = Path(eval_def.data)

        # Check if path is relative to config file
        if not data_path.is_absolute():
            data_path = config_path.parent / data_path

        if not data_path.exists():
            errors.append(f"Data file not found: {eval_def.data}")
            data_files_valid = False
        else:
            try:
                test_data = load_test_data(data_path)
                if not test_data:
                    warnings.append(f"Data file is empty: {eval_def.data}")
                elif len(test_data) == 0:
                    warnings.append(f"No test cases in: {eval_def.data}")
            except Exception as e:
                errors.append(f"Error loading data file {eval_def.data}: {e}")
                data_files_valid = False

    if data_files_valid and not errors:
        console.print("[green]✓[/green] All data files are accessible")

    # 4. Check API keys
    api_key = os.environ.get("FI_API_KEY")
    secret_key = os.environ.get("FI_SECRET_KEY")

    if not api_key:
        warnings.append("FI_API_KEY environment variable not set")
    if not secret_key:
        warnings.append("FI_SECRET_KEY environment variable not set")

    if api_key and secret_key:
        console.print("[green]✓[/green] API keys are configured")
    else:
        console.print("[yellow]![/yellow] API keys not configured (evaluations will fail)")

    # 5. Validate output configuration
    if eval_config.output:
        output_path = Path(eval_config.output.path)
        if not output_path.is_absolute():
            output_path = config_path.parent / output_path

        if output_path.exists() and not output_path.is_dir():
            warnings.append(f"Output path exists but is not a directory: {eval_config.output.path}")
        elif not output_path.exists():
            warnings.append(f"Output directory does not exist (will be created): {eval_config.output.path}")

    # Print results
    _print_validation_results(errors, warnings, strict)

    if errors:
        raise typer.Exit(1)
    elif strict and warnings:
        raise typer.Exit(1)


def _print_validation_results(errors: list, warnings: list, strict: bool) -> None:
    """Print validation errors and warnings."""
    console.print()

    if errors:
        console.print("[bold red]Errors:[/bold red]")
        for error in errors:
            console.print(f"  [red]✗[/red] {error}")

    if warnings:
        console.print("[bold yellow]Warnings:[/bold yellow]")
        for warning in warnings:
            console.print(f"  [yellow]![/yellow] {warning}")

    console.print()

    if errors:
        print_error(f"Validation failed with {len(errors)} error(s)")
    elif warnings and strict:
        print_warning(f"Validation failed with {len(warnings)} warning(s) (strict mode)")
    elif warnings:
        print_warning(f"Validation passed with {len(warnings)} warning(s)")
    else:
        print_success("Validation passed!")
