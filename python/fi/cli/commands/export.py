"""Export command for exporting evaluation results."""

import json
from pathlib import Path
from typing import Optional

import typer

from fi.cli.storage import RunHistory
from fi.cli.utils.console import console, print_error, print_success, print_warning


def export(
    run_id: Optional[str] = typer.Argument(
        None,
        help="Run ID to export (use 'fi view --list' to see available runs)",
    ),
    output: Path = typer.Option(
        ...,
        "--output", "-o",
        help="Output file path",
    ),
    format: str = typer.Option(
        "json",
        "--format", "-f",
        help="Export format: json, csv, html, junit",
    ),
    last: bool = typer.Option(
        False,
        "--last", "-l",
        help="Export the most recent run",
    ),
) -> None:
    """
    Export evaluation results to a file.

    Supported formats:
        - json: JSON format with full details
        - csv: CSV spreadsheet format
        - html: HTML report for viewing in browser
        - junit: JUnit XML format for CI/CD integration

    Examples:
        fi export --last -o results.json
        fi export --last -o report.html -f html
        fi export --last -o results.xml -f junit
        fi export 20260123-143022-abc -o export.csv -f csv
    """
    history = RunHistory()

    # Get the run to export
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
        print_error("Please specify a run ID or use --last to export the most recent run.")
        raise typer.Exit(1)

    # Load full results
    results = history.load_results(record.run_id)
    if results is None:
        print_error(f"Results file not found for run: {record.run_id}")
        raise typer.Exit(1)

    # Export based on format
    format = format.lower()
    if format == "json":
        _export_json(record, results, output)
    elif format == "csv":
        _export_csv(record, results, output)
    elif format == "html":
        _export_html(record, results, output)
    elif format == "junit":
        _export_junit(record, results, output)
    else:
        print_error(f"Unsupported format: {format}\nSupported: json, csv, html, junit")
        raise typer.Exit(1)

    print_success(f"Exported run {record.run_id} to {output}")


def _export_json(record, results: dict, output: Path) -> None:
    """Export results as JSON."""
    export_data = {
        "run_id": record.run_id,
        "timestamp": record.timestamp,
        "config_file": record.config_file,
        "templates": record.templates,
        "summary": {
            "total": record.total_evaluations,
            "successful": record.successful,
            "failed": record.failed,
            "pass_rate": record.pass_rate,
            "avg_score": record.avg_score,
        },
        "eval_results": results.get("eval_results", []),
    }

    with open(output, "w") as f:
        json.dump(export_data, f, indent=2)


def _export_csv(record, results: dict, output: Path) -> None:
    """Export results as CSV."""
    import csv

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "run_id",
            "timestamp",
            "template",
            "output",
            "reason",
            "runtime",
            "output_type",
            "eval_id",
        ])

        # Data rows
        for result in results.get("eval_results", []):
            writer.writerow([
                record.run_id,
                record.timestamp,
                result.get("name", ""),
                result.get("output", ""),
                result.get("reason", ""),
                result.get("runtime", ""),
                result.get("output_type", ""),
                result.get("eval_id", ""),
            ])


def _export_html(record, results: dict, output: Path) -> None:
    """Export results as HTML."""
    from fi.cli.commands.view import _generate_html_report

    html_content = _generate_html_report(record, results)

    with open(output, "w") as f:
        f.write(html_content)


def _export_junit(record, results: dict, output: Path) -> None:
    """Export results as JUnit XML for CI/CD integration."""
    import xml.etree.ElementTree as ET
    from xml.dom import minidom

    # Create testsuites root
    testsuites = ET.Element("testsuites")
    testsuites.set("name", "AI Evaluation Results")
    testsuites.set("tests", str(record.total_evaluations))
    testsuites.set("failures", str(record.failed))
    testsuites.set("time", "0")

    # Create testsuite for this run
    testsuite = ET.SubElement(testsuites, "testsuite")
    testsuite.set("name", f"Run {record.run_id}")
    testsuite.set("tests", str(record.total_evaluations))
    testsuite.set("failures", str(record.failed))
    testsuite.set("timestamp", record.timestamp)

    # Add test cases
    for result in results.get("eval_results", []):
        testcase = ET.SubElement(testsuite, "testcase")
        testcase.set("name", result.get("name", "Unknown"))
        testcase.set("classname", "fi.evals")

        runtime_ms = result.get("runtime", 0)
        runtime_sec = runtime_ms / 1000 if runtime_ms else 0
        testcase.set("time", f"{runtime_sec:.3f}")

        result_output = result.get("output")
        reason = result.get("reason", "")

        # Check if this is a failure
        is_failure = False
        if isinstance(result_output, bool) and result_output is False:
            is_failure = True
        elif isinstance(result_output, (int, float)) and result_output < 0.5:
            # Consider scores below 0.5 as failures for JUnit purposes
            is_failure = True

        if is_failure:
            failure = ET.SubElement(testcase, "failure")
            failure.set("message", f"Evaluation failed: {result_output}")
            failure.text = reason

    # Pretty print XML
    xml_str = ET.tostring(testsuites, encoding="unicode")
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")

    # Remove extra blank lines
    lines = [line for line in pretty_xml.split("\n") if line.strip()]
    pretty_xml = "\n".join(lines)

    with open(output, "w") as f:
        f.write(pretty_xml)
