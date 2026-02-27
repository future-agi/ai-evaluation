"""Run command for executing evaluations."""

import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from fi.cli.config.loader import load_config, load_test_data
from fi.cli.output.formatters import format_results
from fi.cli.output.reporters import ResultReporter
from fi.cli.utils.console import console, print_error, print_success, print_warning
from fi.cli.assertions import (
    AssertionEvaluator,
    AssertionReporter,
    ExitCode,
)


class ExecutionModeOption(str, Enum):
    """Execution mode options for CLI."""
    local = "local"
    cloud = "cloud"
    hybrid = "hybrid"


def run(
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file",
    ),
    eval_template: Optional[str] = typer.Option(
        None,
        "--eval", "-e",
        help="Evaluation template to run (overrides config)",
    ),
    data: Optional[Path] = typer.Option(
        None,
        "--data", "-d",
        help="Path to test data file (overrides config)",
    ),
    output: str = typer.Option(
        "table",
        "--output", "-o",
        help="Output format: table, json, csv, html",
    ),
    parallel: int = typer.Option(
        8,
        "--parallel", "-p",
        help="Number of parallel workers",
    ),
    timeout: int = typer.Option(
        200,
        "--timeout", "-T",
        help="Timeout per evaluation in seconds",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Model for LLM-as-judge evaluations",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config without running evaluations",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file", "-O",
        help="Path to save output file",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Suppress progress output",
    ),
    no_save: bool = typer.Option(
        False,
        "--no-save",
        help="Don't save run to history (for 'fi view' command)",
    ),
    check_assertions: bool = typer.Option(
        True,
        "--check/--no-check",
        help="Check assertions after evaluation (if configured)",
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast",
        help="Stop on first assertion failure",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit with error code on assertion warnings",
    ),
    mode: ExecutionModeOption = typer.Option(
        ExecutionModeOption.cloud,
        "--mode",
        help="Execution mode: local (no API), cloud (API only), or hybrid (auto-route)",
    ),
    local_llm: Optional[str] = typer.Option(
        None,
        "--local-llm",
        help="Local LLM for LLM-based evals (e.g., 'ollama/llama3.2')",
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Run in offline mode (no cloud API calls)",
    ),
) -> None:
    """
    Run evaluations from config file or CLI arguments.

    Examples:
        fi run                              # Use default config (cloud mode)
        fi run -c custom.yaml               # Use custom config
        fi run -e groundedness -d data.json # Run single evaluation
        fi run -o json > results.json       # Output as JSON
        fi run --mode local                 # Run only local heuristic metrics
        fi run --mode hybrid                # Auto-route between local and cloud
        fi run --local-llm ollama/llama3.2  # Use local LLM for LLM-based evals
        fi run --offline                    # No cloud API calls (implies local mode)
    """
    from fi.evals.evaluator import Evaluator
    from fi.evals.local import ExecutionMode, HybridEvaluator, LocalEvaluator

    # Handle offline mode implications
    effective_mode = mode
    if offline:
        if mode == ExecutionModeOption.cloud:
            effective_mode = ExecutionModeOption.local
            print_warning("Offline mode enabled - switching from cloud to local mode")

    # Check for API keys (only required for cloud/hybrid modes)
    api_key = os.environ.get("FI_API_KEY")
    secret_key = os.environ.get("FI_SECRET_KEY")

    if effective_mode != ExecutionModeOption.local and (not api_key or not secret_key):
        if effective_mode == ExecutionModeOption.cloud:
            print_warning(
                "API keys not found in environment.\n"
                "Set FI_API_KEY and FI_SECRET_KEY environment variables."
            )
            if not dry_run:
                raise typer.Exit(1)
        elif effective_mode == ExecutionModeOption.hybrid:
            print_warning(
                "API keys not found - hybrid mode will only run local metrics."
            )

    # Load configuration or use CLI arguments
    if eval_template and data:
        # CLI-only mode
        test_data = load_test_data(data)
        evaluations = [{"template": eval_template, "data": test_data}]
        defaults = {"timeout": timeout, "parallel_workers": parallel}
    else:
        # Config file mode
        try:
            eval_config = load_config(config)
        except FileNotFoundError as e:
            print_error(str(e))
            raise typer.Exit(1)
        except ValueError as e:
            print_error(f"Configuration error: {e}")
            raise typer.Exit(1)

        defaults = {
            "timeout": eval_config.get_defaults().timeout,
            "parallel_workers": eval_config.get_defaults().parallel_workers,
            "model": eval_config.get_defaults().model,
        }

        # Override with CLI arguments
        if timeout != 200:
            defaults["timeout"] = timeout
        if parallel != 8:
            defaults["parallel_workers"] = parallel
        if model:
            defaults["model"] = model

        # Build evaluations list
        evaluations = []
        for eval_def in eval_config.evaluations:
            try:
                test_data = load_test_data(eval_def.data)
            except FileNotFoundError as e:
                print_error(f"Data file not found: {eval_def.data}")
                raise typer.Exit(1)

            templates = eval_def.templates or [eval_def.template]
            for template in templates:
                if template:
                    evaluations.append({
                        "name": eval_def.name,
                        "template": template,
                        "data": test_data,
                        "config": eval_def.config,
                    })

    if dry_run:
        print_success(
            f"Configuration valid!\n"
            f"Would run {len(evaluations)} evaluation(s) in {effective_mode.value} mode."
        )
        return

    # Initialize local LLM if specified
    llm_instance = None
    if local_llm:
        try:
            from fi.evals.local import LocalLLMFactory
            llm_instance = LocalLLMFactory.from_string(local_llm)
            if not llm_instance.is_available():
                print_warning(
                    f"Local LLM '{local_llm}' is not available. "
                    "Make sure Ollama is running: `ollama serve`"
                )
                llm_instance = None
            elif not quiet:
                print_success(f"Local LLM initialized: {local_llm}")
        except Exception as e:
            print_warning(f"Failed to initialize local LLM: {e}")
            llm_instance = None

    # Initialize evaluator(s) based on mode
    cloud_evaluator = None
    hybrid_evaluator = None

    if effective_mode == ExecutionModeOption.cloud:
        # Pure cloud mode - use standard evaluator
        cloud_evaluator = Evaluator(
            fi_api_key=api_key,
            fi_secret_key=secret_key,
            max_workers=defaults["parallel_workers"],
        )
    elif effective_mode == ExecutionModeOption.local:
        # Pure local mode - use hybrid evaluator in offline mode
        hybrid_evaluator = HybridEvaluator(
            local_llm=llm_instance,
            prefer_local=True,
            fallback_to_cloud=False,
            offline_mode=True,
        )
        if not quiet:
            print_success("Running in local mode (no cloud API calls)")
    else:
        # Hybrid mode - use hybrid evaluator with cloud fallback
        if api_key and secret_key:
            cloud_evaluator = Evaluator(
                fi_api_key=api_key,
                fi_secret_key=secret_key,
                max_workers=defaults["parallel_workers"],
            )
        hybrid_evaluator = HybridEvaluator(
            local_llm=llm_instance,
            cloud_evaluator=cloud_evaluator,
            prefer_local=True,
            fallback_to_cloud=not offline,
            offline_mode=offline,
        )
        if not quiet:
            print_success("Running in hybrid mode (auto-routing local/cloud)")

    all_results = []
    local_metrics_run = 0
    cloud_metrics_run = 0

    # Helper function to run single evaluation
    def run_evaluation(eval_def: Dict[str, Any]) -> None:
        nonlocal local_metrics_run, cloud_metrics_run

        template = eval_def["template"]
        data = eval_def["data"]

        if effective_mode == ExecutionModeOption.cloud:
            # Pure cloud mode
            results = cloud_evaluator.evaluate(
                eval_templates=template,
                inputs=data,
                timeout=defaults["timeout"],
                model_name=defaults.get("model"),
            )
            all_results.extend(results.eval_results)
            cloud_metrics_run += 1

        elif effective_mode == ExecutionModeOption.local:
            # Pure local mode
            result = hybrid_evaluator.evaluate(
                template=template,
                inputs=data,
                config=eval_def.get("config", {}),
            )
            all_results.extend(result.results.eval_results)
            local_metrics_run += len(result.executed_locally)

        else:
            # Hybrid mode - route based on metric type
            from fi.evals.local import can_run_locally

            if can_run_locally(template) or hybrid_evaluator.can_use_local_llm(template):
                # Run locally
                result = hybrid_evaluator.evaluate(
                    template=template,
                    inputs=data,
                    config=eval_def.get("config", {}),
                )
                all_results.extend(result.results.eval_results)
                local_metrics_run += len(result.executed_locally)
            elif cloud_evaluator:
                # Run in cloud
                results = cloud_evaluator.evaluate(
                    eval_templates=template,
                    inputs=data,
                    timeout=defaults["timeout"],
                    model_name=defaults.get("model"),
                )
                all_results.extend(results.eval_results)
                cloud_metrics_run += 1
            else:
                # No cloud available
                from fi.evals.types import EvalResult
                for _ in data:
                    all_results.append(
                        EvalResult(
                            name=template,
                            output=None,
                            reason="Cloud unavailable - cannot run this metric locally",
                            runtime=0,
                        )
                    )

    # Run evaluations
    if not quiet:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for eval_def in evaluations:
                task = progress.add_task(
                    f"Running {eval_def['template']}...",
                    total=None
                )

                try:
                    run_evaluation(eval_def)
                    progress.update(task, description=f"[green]✓[/green] {eval_def['template']}")
                except Exception as e:
                    progress.update(task, description=f"[red]✗[/red] {eval_def['template']}: {e}")
                    if not quiet:
                        console.print(f"[red]Error running {eval_def['template']}: {e}[/red]")

                progress.remove_task(task)
    else:
        for eval_def in evaluations:
            try:
                run_evaluation(eval_def)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]", file=sys.stderr)

    # Show execution summary in hybrid mode
    if not quiet and effective_mode == ExecutionModeOption.hybrid:
        console.print(
            f"\n[dim]Executed: {local_metrics_run} local, {cloud_metrics_run} cloud[/dim]"
        )

    # Create combined results
    from fi.evals.types import BatchRunResult
    combined_results = BatchRunResult(eval_results=all_results)

    # Format and display results
    output_path = str(output_file) if output_file else None
    result_str = format_results(combined_results, output, console, output_path)

    # For non-table formats, print the result
    if output != "table" and result_str:
        if output_file:
            print_success(f"Results saved to: {output_file}")
        else:
            console.print(result_str)

    # Print summary for table output
    if output == "table" and not quiet:
        reporter = ResultReporter(console)
        reporter.report_summary(combined_results)

    # Save run to history
    if not no_save and all_results:
        from fi.cli.storage import RunHistory

        history = RunHistory()
        templates_used = list(set(e["template"] for e in evaluations))
        config_path_str = str(config) if config else None

        record = history.save_run(
            results=combined_results,
            config_file=config_path_str,
            templates=templates_used,
        )

        if not quiet:
            console.print(f"\n[dim]Run saved: {record.run_id}[/dim]")
            console.print("[dim]View with: fi view --last[/dim]")

    # Check assertions if configured
    if check_assertions and 'eval_config' in dir() and eval_config is not None:
        assertion_config = _build_assertion_config(eval_config, fail_fast)

        if assertion_config.get('assertions') or assertion_config.get('thresholds', {}).get('default_pass_rate'):
            # Convert results to dict format for evaluator
            results_dict = {
                "eval_results": [
                    {
                        "name": r.name,
                        "output": r.output,
                        "reason": r.reason,
                        "runtime": r.runtime,
                        "output_type": r.output_type,
                        "eval_id": r.eval_id,
                    }
                    for r in all_results
                ]
            }

            evaluator = AssertionEvaluator(results_dict, assertion_config)
            report = evaluator.evaluate_all()

            # Display assertion report
            if not quiet and report.total_assertions > 0:
                reporter = AssertionReporter(console)
                console.print()  # Blank line before assertions
                reporter.display(report)
                reporter.display_summary_line(report)

            # Determine exit code based on assertion results
            if report.failed > 0:
                raise typer.Exit(ExitCode.ASSERTION_FAILED)
            elif report.warnings > 0 and strict:
                raise typer.Exit(ExitCode.ASSERTION_WARNING)


def _build_assertion_config(
    eval_config,
    fail_fast: bool = False
) -> Dict[str, Any]:
    """Build assertion config dictionary from FIEvaluationConfig.

    Args:
        eval_config: The loaded FIEvaluationConfig object.
        fail_fast: Whether to enable fail-fast mode.

    Returns:
        Dictionary with 'assertions' and 'thresholds' keys.
    """
    assertion_config: Dict[str, Any] = {
        "assertions": [],
        "thresholds": {}
    }

    # Convert assertion configs
    if eval_config.assertions:
        for assertion in eval_config.assertions:
            assertion_dict = {
                "template": assertion.template,
                "global": assertion.is_global,
                "conditions": assertion.conditions,
                "on_fail": assertion.on_fail,
            }
            assertion_config["assertions"].append(assertion_dict)

    # Convert thresholds config
    if eval_config.thresholds:
        assertion_config["thresholds"] = {
            "default_pass_rate": eval_config.thresholds.default_pass_rate,
            "fail_fast": fail_fast or eval_config.thresholds.fail_fast,
            "overrides": eval_config.thresholds.overrides or {},
        }

    return assertion_config
