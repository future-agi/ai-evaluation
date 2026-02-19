"""Console utilities for CLI output."""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Global console instance
console = Console()


def print_error(message: str, title: str = "Error") -> None:
    """Print an error message with red styling."""
    console.print(Panel(
        Text(message, style="red"),
        title=f"[red]{title}[/red]",
        border_style="red"
    ))


def print_success(message: str, title: str = "Success") -> None:
    """Print a success message with green styling."""
    console.print(Panel(
        Text(message, style="green"),
        title=f"[green]{title}[/green]",
        border_style="green"
    ))


def print_warning(message: str, title: str = "Warning") -> None:
    """Print a warning message with yellow styling."""
    console.print(Panel(
        Text(message, style="yellow"),
        title=f"[yellow]{title}[/yellow]",
        border_style="yellow"
    ))


def print_info(message: str, title: str = "Info") -> None:
    """Print an info message with blue styling."""
    console.print(Panel(
        Text(message, style="blue"),
        title=f"[blue]{title}[/blue]",
        border_style="blue"
    ))
