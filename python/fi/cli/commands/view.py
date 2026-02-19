"""View command for displaying evaluation results."""

import json
import tempfile
import webbrowser
from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel
from rich.table import Table

from fi.cli.storage import RunHistory, RunRecord
from fi.cli.utils.console import console, print_error, print_success, print_warning


def view(
    run_id: Optional[str] = typer.Argument(
        None,
        help="Run ID to view (use 'fi view --list' to see available runs)",
    ),
    last: bool = typer.Option(
        False,
        "--last", "-l",
        help="View the most recent run",
    ),
    list_runs: bool = typer.Option(
        False,
        "--list",
        help="List recent runs",
    ),
    terminal: bool = typer.Option(
        False,
        "--terminal", "-t",
        help="Display in terminal instead of browser",
    ),
    limit: int = typer.Option(
        10,
        "--limit", "-n",
        help="Number of runs to list (with --list)",
    ),
    detailed: bool = typer.Option(
        False,
        "--detailed", "-d",
        help="Show detailed results in terminal mode",
    ),
) -> None:
    """
    View evaluation results from previous runs.

    Examples:
        fi view --list              # List recent runs
        fi view --last              # View most recent run in browser
        fi view --last -t           # View most recent run in terminal
        fi view 20260123-143022-abc # View specific run by ID
    """
    history = RunHistory()

    if list_runs:
        _list_runs(history, limit)
        return

    # Get the run to view
    if last:
        record = history.get_latest_run()
        if record is None:
            print_warning("No runs found. Run 'fi run' first to create evaluations.")
            raise typer.Exit(1)
    elif run_id:
        record = history.get_run(run_id)
        if record is None:
            print_error(f"Run not found: {run_id}\nUse 'fi view --list' to see available runs.")
            raise typer.Exit(1)
    else:
        # No arguments - show list
        print_warning("Please specify a run ID or use --last to view the most recent run.")
        console.print("\n[dim]Use 'fi view --list' to see available runs.[/dim]")
        raise typer.Exit(1)

    # Load full results
    results = history.load_results(record.run_id)
    if results is None:
        print_error(f"Results file not found for run: {record.run_id}")
        raise typer.Exit(1)

    if terminal:
        _display_terminal(record, results, detailed)
    else:
        _display_browser(record, results)


def _list_runs(history: RunHistory, limit: int) -> None:
    """List recent runs in a table."""
    runs = history.list_runs(limit)

    if not runs:
        print_warning("No runs found. Run 'fi run' to create evaluations.")
        return

    table = Table(
        title=f"Recent Evaluation Runs ({len(runs)} shown)",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Run ID", style="green")
    table.add_column("Timestamp", style="dim")
    table.add_column("Templates", style="blue")
    table.add_column("Total", justify="right")
    table.add_column("Pass Rate", justify="right")

    for run in runs:
        templates_str = ", ".join(run.templates[:3])
        if len(run.templates) > 3:
            templates_str += f" (+{len(run.templates) - 3} more)"

        pass_rate_str = f"{run.pass_rate:.1f}%" if run.pass_rate is not None else "-"

        table.add_row(
            run.run_id,
            run.timestamp[:19],  # Truncate to seconds
            templates_str,
            str(run.total_evaluations),
            pass_rate_str,
        )

    console.print(table)
    console.print("\n[dim]Use 'fi view <run_id>' to view details.[/dim]")


def _display_terminal(record: RunRecord, results: dict, detailed: bool) -> None:
    """Display results in terminal."""
    # Summary panel
    summary_parts = [
        f"[bold]Run ID:[/bold] {record.run_id}",
        f"[bold]Timestamp:[/bold] {record.timestamp}",
        f"[bold]Templates:[/bold] {', '.join(record.templates)}",
        f"[bold]Total:[/bold] {record.total_evaluations}",
        f"[bold]Successful:[/bold] [green]{record.successful}[/green]",
    ]

    if record.failed > 0:
        summary_parts.append(f"[bold]Failed:[/bold] [red]{record.failed}[/red]")

    if record.pass_rate is not None:
        summary_parts.append(f"[bold]Pass Rate:[/bold] {record.pass_rate:.1f}%")

    if record.avg_score is not None:
        summary_parts.append(f"[bold]Avg Score:[/bold] {record.avg_score:.3f}")

    panel = Panel(
        "\n".join(summary_parts),
        title="Run Summary",
        border_style="blue",
    )
    console.print(panel)

    if detailed:
        # Results table
        table = Table(
            title="Evaluation Results",
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Template", style="green")
        table.add_column("Output", style="white")
        table.add_column("Reason", style="dim", max_width=50)
        table.add_column("Runtime", justify="right", style="blue")

        for result in results.get("eval_results", []):
            output_str = str(result.get("output", "N/A"))
            reason_str = result.get("reason") or "N/A"

            # Truncate long strings
            if len(reason_str) > 50:
                reason_str = reason_str[:47] + "..."

            table.add_row(
                result.get("name", "Unknown"),
                output_str,
                reason_str,
                str(result.get("runtime", "N/A")),
            )

        console.print(table)
    else:
        console.print("\n[dim]Use --detailed to see full results table.[/dim]")


def _display_browser(record: RunRecord, results: dict) -> None:
    """Display results in browser."""
    html_content = _generate_html_report(record, results)

    # Write to temp file
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".html",
        delete=False,
        prefix=f"fi-eval-{record.run_id}-",
    ) as f:
        f.write(html_content)
        temp_path = f.name

    # Open in browser
    console.print(f"[dim]Opening results in browser: {temp_path}[/dim]")
    webbrowser.open(f"file://{temp_path}")
    print_success(f"Opened run {record.run_id} in browser")


def _generate_html_report(record: RunRecord, results: dict) -> str:
    """Generate an HTML report for the results."""
    html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Results - {run_id}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        .subtitle {{
            color: #666;
            margin-bottom: 20px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .summary-card .label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .summary-card.success .value {{ color: #28a745; }}
        .summary-card.danger .value {{ color: #dc3545; }}
        .summary-card.info .value {{ color: #007bff; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .output-true {{ color: #28a745; font-weight: bold; }}
        .output-false {{ color: #dc3545; font-weight: bold; }}
        .output-number {{ color: #007bff; font-weight: bold; }}
        .reason {{
            max-width: 400px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .footer {{
            margin-top: 30px;
            text-align: center;
            color: #999;
            font-size: 0.9em;
        }}
        .templates {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-bottom: 20px;
        }}
        .template-tag {{
            background: #e9ecef;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.85em;
            color: #495057;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Evaluation Results</h1>
        <p class="subtitle">Run ID: {run_id} | {timestamp}</p>

        <div class="templates">
            {template_tags}
        </div>

        <div class="summary">
            <div class="summary-card">
                <div class="value">{total}</div>
                <div class="label">Total Evaluations</div>
            </div>
            <div class="summary-card success">
                <div class="value">{successful}</div>
                <div class="label">Successful</div>
            </div>
            {failed_card}
            {pass_rate_card}
            {avg_score_card}
        </div>

        <table>
            <thead>
                <tr>
                    <th>Template</th>
                    <th>Output</th>
                    <th>Reason</th>
                    <th>Runtime (ms)</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>

        <div class="footer">
            Generated by Future AGI Evaluation CLI
        </div>
    </div>
</body>
</html>"""

    # Generate template tags
    template_tags = "\n".join(
        f'<span class="template-tag">{t}</span>' for t in record.templates
    )

    # Generate result rows
    rows = []
    for result in results.get("eval_results", []):
        output = result.get("output")
        if isinstance(output, bool):
            output_class = "output-true" if output else "output-false"
            output_str = "Pass" if output else "Fail"
        elif isinstance(output, (int, float)):
            output_class = "output-number"
            output_str = f"{output:.3f}" if isinstance(output, float) else str(output)
        else:
            output_class = ""
            output_str = str(output)

        reason = result.get("reason") or "-"

        rows.append(f"""
            <tr>
                <td>{result.get("name", "Unknown")}</td>
                <td class="{output_class}">{output_str}</td>
                <td class="reason" title="{reason}">{reason}</td>
                <td>{result.get("runtime", "-")}</td>
            </tr>
        """)

    # Optional cards
    failed_card = ""
    if record.failed > 0:
        failed_card = f"""
            <div class="summary-card danger">
                <div class="value">{record.failed}</div>
                <div class="label">Failed</div>
            </div>
        """

    pass_rate_card = ""
    if record.pass_rate is not None:
        pass_rate_card = f"""
            <div class="summary-card info">
                <div class="value">{record.pass_rate:.1f}%</div>
                <div class="label">Pass Rate</div>
            </div>
        """

    avg_score_card = ""
    if record.avg_score is not None:
        avg_score_card = f"""
            <div class="summary-card info">
                <div class="value">{record.avg_score:.3f}</div>
                <div class="label">Avg Score</div>
            </div>
        """

    return html_template.format(
        run_id=record.run_id,
        timestamp=record.timestamp[:19],
        template_tags=template_tags,
        total=record.total_evaluations,
        successful=record.successful,
        failed_card=failed_card,
        pass_rate_card=pass_rate_card,
        avg_score_card=avg_score_card,
        rows="\n".join(rows),
    )
