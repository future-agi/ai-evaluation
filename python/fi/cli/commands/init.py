"""Init command for creating new evaluation projects."""

import os
from pathlib import Path
from typing import Optional

import typer

from fi.cli.config.defaults import TEMPLATES, SAMPLE_DATA
from fi.cli.utils.console import console, print_success, print_error, print_warning


def init_project(
    directory: Optional[Path] = typer.Argument(
        None,
        help="Directory to initialize (default: current directory)",
    ),
    template: str = typer.Option(
        "basic",
        "--template", "-t",
        help="Template to use: basic, rag, safety, agent",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Overwrite existing configuration file",
    ),
) -> None:
    """
    Initialize a new AI evaluation project.

    Creates a fi-evaluation.yaml configuration file and sample data files.
    """
    # Determine target directory
    if directory is None:
        target_dir = Path.cwd()
    else:
        target_dir = Path(directory).resolve()

    # Validate template
    if template not in TEMPLATES:
        print_error(
            f"Unknown template: {template}\n"
            f"Available templates: {', '.join(TEMPLATES.keys())}"
        )
        raise typer.Exit(1)

    # Create directory if it doesn't exist
    if not target_dir.exists():
        target_dir.mkdir(parents=True)
        console.print(f"[dim]Created directory: {target_dir}[/dim]")

    # Check for existing config
    config_path = target_dir / "fi-evaluation.yaml"
    if config_path.exists() and not force:
        print_warning(
            f"Configuration file already exists: {config_path}\n"
            "Use --force to overwrite."
        )
        raise typer.Exit(1)

    # Create data directory
    data_dir = target_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Create results directory
    results_dir = target_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Write configuration file
    config_content = TEMPLATES[template]
    with open(config_path, "w") as f:
        f.write(config_content)
    console.print(f"[dim]Created: {config_path}[/dim]")

    # Write sample data
    sample_data = SAMPLE_DATA.get(template, SAMPLE_DATA["basic"])
    data_filename = _get_data_filename(template)
    data_path = data_dir / data_filename
    with open(data_path, "w") as f:
        f.write(sample_data)
    console.print(f"[dim]Created: {data_path}[/dim]")

    # Create .gitignore for results
    gitignore_path = results_dir / ".gitignore"
    with open(gitignore_path, "w") as f:
        f.write("# Ignore evaluation results\n*\n!.gitignore\n")

    # Success message
    print_success(
        f"Initialized evaluation project with '{template}' template!\n\n"
        f"Project structure:\n"
        f"  {target_dir}/\n"
        f"  ├── fi-evaluation.yaml  # Configuration file\n"
        f"  ├── data/\n"
        f"  │   └── {data_filename}  # Sample test data\n"
        f"  └── results/            # Evaluation results\n\n"
        f"Next steps:\n"
        f"  1. Edit fi-evaluation.yaml to configure your evaluations\n"
        f"  2. Add your test data to the data/ directory\n"
        f"  3. Run 'fi run' to execute evaluations"
    )


def _get_data_filename(template: str) -> str:
    """Get the appropriate data filename for the template."""
    filenames = {
        "basic": "test_cases.json",
        "rag": "rag_test_cases.json",
        "safety": "safety_test_cases.json",
        "agent": "agent_test_cases.json",
    }
    return filenames.get(template, "test_cases.json")
