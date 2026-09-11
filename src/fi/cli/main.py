"""
AI Evaluation CLI

Command-line interface for running LLM evaluations with 60+ templates.
"""

import typer
from rich.console import Console

from fi.cli.commands.init import init_project
from fi.cli.commands.run import run
from fi.cli.commands.list_cmd import list_resources
from fi.cli.commands.validate import validate
from fi.cli.commands.config import config_app
from fi.cli.commands.view import view
from fi.cli.commands.export import export

# Create main app
app = typer.Typer(
    name="fi",
    help="AI Evaluation CLI - Evaluate LLM outputs with 60+ templates",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()


# Register commands
app.command("init", help="Initialize a new evaluation project")(init_project)
app.command("run", help="Run evaluations from config or CLI")(run)
app.command("list", help="List available templates and resources")(list_resources)
app.command("validate", help="Validate configuration file")(validate)
app.command("view", help="View evaluation results from previous runs")(view)
app.command("export", help="Export evaluation results to file")(export)
app.add_typer(config_app, name="config", help="Manage CLI configuration")


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version", "-v",
        help="Show version and exit",
    ),
) -> None:
    """
    AI Evaluation CLI by Future AGI.

    Evaluate LLM outputs with 60+ pre-built evaluation templates.

    Quick Start:
        fi init my-project      # Initialize project
        fi run                  # Run evaluations
        fi list templates       # List available templates
    """
    if version:
        from importlib.metadata import version as get_version
        try:
            v = get_version("ai-evaluation")
        except Exception:
            v = "1.0.0"
        console.print(f"ai-evaluation version {v}")
        raise typer.Exit()

    # If no command provided and not asking for version, show help
    if ctx.invoked_subcommand is None and not version:
        console.print(ctx.get_help())


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
