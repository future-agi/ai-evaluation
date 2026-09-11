"""List command for displaying available resources."""

import json
from typing import Optional

import typer
from rich.table import Table

from fi.cli.utils.console import console, print_error
from fi.evals.templates import EvalTemplate


# Template categories for filtering
TEMPLATE_CATEGORIES = {
    "conversation": [
        "conversation_coherence",
        "conversation_resolution",
    ],
    "rag": [
        "context_adherence",
        "context_relevance",
        "completeness",
        "chunk_attribution",
        "chunk_utilization",
        "groundedness",
    ],
    "safety": [
        "content_moderation",
        "pii",
        "toxicity",
        "prompt_injection",
        "content_safety_violation",
        "no_harmful_therapeutic_guidance",
        "is_harmful_advice",
    ],
    "bias": [
        "no_racial_bias",
        "no_gender_bias",
        "no_age_bias",
        "bias_detection",
        "sexist",
    ],
    "quality": [
        "factual_accuracy",
        "summary_quality",
        "is_good_summary",
        "is_factually_consistent",
        "completeness",
    ],
    "format": [
        "is_json",
        "is_csv",
        "is_code",
        "is_email",
        "one_line",
        "contains_valid_link",
        "no_valid_links",
    ],
    "tone": [
        "tone",
        "is_polite",
        "is_concise",
        "is_helpful",
        "is_informal_tone",
        "clinically_inappropriate_tone",
    ],
    "translation": [
        "translation_accuracy",
        "cultural_sensitivity",
    ],
    "audio": [
        "audio_transcription",
        "audio_quality",
    ],
    "function_calling": [
        "llm_function_calling",
        "evaluate_function_calling",
    ],
    "hallucination": [
        "detect_hallucination_missing_info",
        "caption_hallucination",
    ],
}


def list_resources(
    resource: str = typer.Argument(
        "templates",
        help="Resource to list: templates, categories",
    ),
    format: str = typer.Option(
        "table",
        "--format", "-f",
        help="Output format: table, json",
    ),
    category: Optional[str] = typer.Option(
        None,
        "--category", "-c",
        help="Filter templates by category",
    ),
) -> None:
    """
    List available evaluation resources.

    Resources:
        templates - List all available evaluation templates
        categories - List template categories
    """
    if resource == "templates":
        _list_templates(format, category)
    elif resource == "categories":
        _list_categories(format)
    else:
        print_error(f"Unknown resource: {resource}\nAvailable: templates, categories")
        raise typer.Exit(1)


def _list_templates(format: str, category: Optional[str] = None) -> None:
    """List all available evaluation templates."""
    # Get all template classes from templates module
    from fi.evals import templates as templates_module

    template_list = []
    for name in dir(templates_module):
        obj = getattr(templates_module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, EvalTemplate)
            and obj is not EvalTemplate
            and hasattr(obj, "eval_name")
        ):
            eval_name = obj.eval_name
            # Filter by category if specified
            if category:
                if category not in TEMPLATE_CATEGORIES:
                    print_error(
                        f"Unknown category: {category}\n"
                        f"Available: {', '.join(TEMPLATE_CATEGORIES.keys())}"
                    )
                    raise typer.Exit(1)
                if eval_name not in TEMPLATE_CATEGORIES[category]:
                    continue

            # Determine category for template
            template_category = _get_template_category(eval_name)

            template_list.append({
                "name": eval_name,
                "class": name,
                "category": template_category,
            })

    # Sort by name
    template_list.sort(key=lambda x: x["name"])

    if format == "json":
        console.print(json.dumps(template_list, indent=2))
    else:
        _print_templates_table(template_list)


def _list_categories(format: str) -> None:
    """List all template categories."""
    categories = [
        {"name": cat, "count": len(templates)}
        for cat, templates in TEMPLATE_CATEGORIES.items()
    ]

    if format == "json":
        console.print(json.dumps(categories, indent=2))
    else:
        table = Table(
            title="Template Categories",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Category", style="green")
        table.add_column("Templates", justify="right", style="blue")

        for cat in categories:
            table.add_row(cat["name"], str(cat["count"]))

        console.print(table)


def _print_templates_table(templates: list) -> None:
    """Print templates as a Rich table."""
    table = Table(
        title=f"Available Evaluation Templates ({len(templates)} total)",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Template Name", style="green")
    table.add_column("Category", style="blue")
    table.add_column("Class", style="dim")

    for template in templates:
        table.add_row(
            template["name"],
            template["category"],
            template["class"],
        )

    console.print(table)


def _get_template_category(eval_name: str) -> str:
    """Get the category for a template."""
    for category, templates in TEMPLATE_CATEGORIES.items():
        if eval_name in templates:
            return category
    return "other"
