"""Summary reporters for CLI evaluation results."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


class ResultReporter:
    """Print a summary panel after evaluation results are displayed."""

    def __init__(self, console: Console):
        self.console = console

    def report_summary(self, batch_result) -> None:
        """Print a summary of the batch evaluation results."""
        results = [r for r in batch_result.eval_results if r is not None]
        total = len(results)

        if total == 0:
            self.console.print("[yellow]No results to summarise.[/yellow]")
            return

        # Count numeric scores (output that can be interpreted as a number)
        scores = []
        for r in results:
            try:
                scores.append(float(r.output))
            except (TypeError, ValueError):
                pass

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("label", style="bold")
        table.add_column("value")

        table.add_row("Total metrics", str(total))

        if scores:
            avg = sum(scores) / len(scores)
            table.add_row("Avg score", f"{avg:.3f}")
            table.add_row("Min score", f"{min(scores):.3f}")
            table.add_row("Max score", f"{max(scores):.3f}")

        total_runtime = sum(r.runtime for r in results)
        table.add_row("Total runtime", f"{total_runtime} ms")

        self.console.print(Panel(table, title="Summary", border_style="blue"))
