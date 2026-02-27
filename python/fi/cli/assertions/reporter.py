"""Reporter for displaying assertion results."""

from typing import Dict, Any, List
import xml.etree.ElementTree as ET
from xml.dom import minidom

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .evaluator import AssertionReport, AssertionResult, AssertionOutcome


class AssertionReporter:
    """Display assertion results in various formats."""

    def __init__(self, console: Console):
        """Initialize the reporter.

        Args:
            console: Rich Console instance for output.
        """
        self.console = console

    def display(self, report: AssertionReport, detailed: bool = True) -> None:
        """Display assertion report to console.

        Args:
            report: The assertion report to display.
            detailed: If True, show detailed table of all assertions.
        """
        if not report.outcomes:
            self.console.print("[dim]No assertions defined[/dim]")
            return

        # Summary panel
        summary = self._build_summary(report)
        self.console.print(Panel(summary, title="Assertion Results", border_style="blue"))

        # Detailed table
        if detailed:
            table = self._build_table(report)
            self.console.print(table)

    def _build_summary(self, report: AssertionReport) -> Text:
        """Build summary text.

        Args:
            report: The assertion report.

        Returns:
            Rich Text object with formatted summary.
        """
        text = Text()
        text.append(f"Total: {report.total_assertions}  ")
        text.append(f"Passed: {report.passed}", style="green")
        text.append("  ")
        if report.failed > 0:
            text.append(f"Failed: {report.failed}", style="red bold")
        else:
            text.append(f"Failed: {report.failed}", style="dim")
        text.append("  ")
        if report.warnings > 0:
            text.append(f"Warnings: {report.warnings}", style="yellow")
        else:
            text.append(f"Warnings: {report.warnings}", style="dim")
        if report.skipped > 0:
            text.append("  ")
            text.append(f"Skipped: {report.skipped}", style="dim")
        return text

    def _build_table(self, report: AssertionReport) -> Table:
        """Build detailed results table.

        Args:
            report: The assertion report.

        Returns:
            Rich Table with assertion details.
        """
        table = Table(show_header=True, header_style="bold")
        table.add_column("Template", style="cyan")
        table.add_column("Condition")
        table.add_column("Expected")
        table.add_column("Actual", justify="right")
        table.add_column("Result")

        result_style = {
            AssertionResult.PASSED: "[green]✓ PASS[/green]",
            AssertionResult.FAILED: "[red]✗ FAIL[/red]",
            AssertionResult.WARNING: "[yellow]⚠ WARN[/yellow]",
            AssertionResult.SKIPPED: "[dim]○ SKIP[/dim]",
        }

        for outcome in report.outcomes:
            table.add_row(
                outcome.template or "[global]",
                outcome.condition,
                outcome.expected,
                f"{outcome.actual:.4f}",
                result_style[outcome.result]
            )

        return table

    def to_json(self, report: AssertionReport) -> Dict[str, Any]:
        """Convert report to JSON-serializable dict.

        Args:
            report: The assertion report.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        return {
            "summary": {
                "total": report.total_assertions,
                "passed": report.passed,
                "failed": report.failed,
                "warnings": report.warnings,
                "skipped": report.skipped,
                "all_passed": report.all_passed,
            },
            "assertions": [
                {
                    "template": o.template,
                    "condition": o.condition,
                    "expected": o.expected,
                    "actual": o.actual,
                    "result": o.result.value,
                    "message": o.message,
                }
                for o in report.outcomes
            ]
        }

    def to_junit(self, report: AssertionReport) -> str:
        """Convert report to JUnit XML format for CI/CD integration.

        Args:
            report: The assertion report.

        Returns:
            JUnit XML string.
        """
        testsuites = ET.Element("testsuites")
        testsuites.set("name", "Assertions")
        testsuites.set("tests", str(report.total_assertions))
        testsuites.set("failures", str(report.failed))

        testsuite = ET.SubElement(testsuites, "testsuite")
        testsuite.set("name", "Evaluation Assertions")
        testsuite.set("tests", str(report.total_assertions))
        testsuite.set("failures", str(report.failed))

        for outcome in report.outcomes:
            testcase = ET.SubElement(testsuite, "testcase")
            testcase.set("name", f"{outcome.template or 'global'}: {outcome.condition}")
            testcase.set("classname", "fi.assertions")

            if outcome.result == AssertionResult.FAILED:
                failure = ET.SubElement(testcase, "failure")
                failure.set("message", outcome.message)
                failure.text = f"Expected: {outcome.expected}, Actual: {outcome.actual}"
            elif outcome.result == AssertionResult.SKIPPED:
                ET.SubElement(testcase, "skipped")

        xml_str = ET.tostring(testsuites, encoding="unicode")
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ")

        # Remove extra blank lines
        lines = [line for line in pretty_xml.split("\n") if line.strip()]
        return "\n".join(lines)

    def display_summary_line(self, report: AssertionReport) -> None:
        """Display a single-line summary suitable for end of run output.

        Args:
            report: The assertion report.
        """
        if report.failed > 0:
            self.console.print(
                f"\n[red]✗ {report.failed} assertion(s) failed[/red]"
            )
        elif report.warnings > 0:
            self.console.print(
                f"\n[yellow]⚠ {report.passed} assertion(s) passed with {report.warnings} warning(s)[/yellow]"
            )
        else:
            self.console.print(
                f"\n[green]✓ All {report.passed} assertion(s) passed[/green]"
            )
