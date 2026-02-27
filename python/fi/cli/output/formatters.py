"""Format evaluation results for CLI output."""

import csv
import io
import json
from typing import Optional

from rich.console import Console
from rich.table import Table


def format_results(
    batch_result,
    fmt: str,
    console: Console,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Format BatchRunResult for display or file output.

    Args:
        batch_result: BatchRunResult with eval_results list.
        fmt: Output format — "table", "json", "csv", or "html".
        console: Rich Console for table rendering.
        output_path: Optional file path to write output to.

    Returns:
        Formatted string for non-table formats, None for table format.
    """
    results = [r for r in batch_result.eval_results if r is not None]

    if fmt == "table":
        _print_table(results, console)
        return None
    elif fmt == "json":
        text = _to_json(results)
    elif fmt == "csv":
        text = _to_csv(results)
    elif fmt == "html":
        text = _to_html(results)
    else:
        text = _to_json(results)

    if output_path:
        with open(output_path, "w") as f:
            f.write(text)

    return text


def _print_table(results, console: Console) -> None:
    table = Table(title="Evaluation Results", show_lines=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Output", style="green")
    table.add_column("Reason", style="dim")
    table.add_column("Runtime (ms)", justify="right")

    for r in results:
        table.add_row(
            r.name,
            str(r.output) if r.output is not None else "-",
            (r.reason or "-")[:80],
            str(r.runtime),
        )

    console.print(table)


def _to_json(results) -> str:
    rows = []
    for r in results:
        rows.append({
            "name": r.name,
            "output": r.output,
            "reason": r.reason,
            "runtime": r.runtime,
        })
    return json.dumps(rows, indent=2, default=str)


def _to_csv(results) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["name", "output", "reason", "runtime"])
    for r in results:
        writer.writerow([r.name, r.output, r.reason, r.runtime])
    return buf.getvalue()


def _to_html(results) -> str:
    rows_html = ""
    for r in results:
        rows_html += (
            f"<tr><td>{r.name}</td><td>{r.output}</td>"
            f"<td>{r.reason or ''}</td><td>{r.runtime}</td></tr>\n"
        )
    return f"""<!DOCTYPE html>
<html><head><title>Evaluation Results</title>
<style>
  body {{ font-family: sans-serif; margin: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
  th {{ background: #f5f5f5; }}
</style></head><body>
<h1>Evaluation Results</h1>
<table><tr><th>Metric</th><th>Output</th><th>Reason</th><th>Runtime (ms)</th></tr>
{rows_html}</table></body></html>"""
