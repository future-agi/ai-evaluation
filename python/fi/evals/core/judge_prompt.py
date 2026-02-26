"""
Generic LLM judge prompt builder for augmenting local metric results.

When a local metric supports LLM augmentation (supports_llm_judge = True)
and the user passes augment=True, the local heuristic runs first, then
this module builds a prompt that feeds the heuristic scores + reasoning
to the LLM for refinement.
"""

import json
from typing import Any, Dict

from .result import EvalResult


_PROMPT_TEMPLATE = """\
You are an expert AI evaluator. Your task is to evaluate: **{metric_name}**

## What this metric measures
{description}

## Local analysis (heuristic pre-screening)
The following analysis was produced by a fast, deterministic heuristic.
Use it as a starting point — it may be accurate, but it cannot reason
about semantics the way you can.

Score: {local_score}
Reasoning: {local_reason}

## Raw data
{formatted_inputs}

## Instructions
Using the local analysis as a starting point and the raw data for verification,
provide your refined judgment.

- If the heuristic score seems correct, confirm it with your own reasoning.
- If you find the heuristic missed something or was too harsh/lenient, adjust.
- Score from 0.0 (worst) to 1.0 (best).

Return ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}\
"""


def _format_inputs(inputs: Dict[str, Any]) -> str:
    """Format evaluation inputs for the prompt, keeping it concise."""
    parts = []
    for key, value in inputs.items():
        if value is None:
            continue
        if isinstance(value, str):
            display = value if len(value) <= 1000 else value[:1000] + "..."
            parts.append(f"**{key}**:\n{display}")
        elif isinstance(value, (list, dict)):
            try:
                dumped = json.dumps(value, indent=2, default=str)
                if len(dumped) > 1500:
                    dumped = dumped[:1500] + "\n..."
                parts.append(f"**{key}**:\n```json\n{dumped}\n```")
            except (TypeError, ValueError):
                parts.append(f"**{key}**: {str(value)[:500]}")
        else:
            parts.append(f"**{key}**: {value}")
    return "\n\n".join(parts) if parts else "(no inputs)"


def build_judge_prompt(
    metric_name: str,
    description: str,
    inputs: Dict[str, Any],
    local_result: EvalResult,
) -> str:
    """Build an LLM judge prompt that includes local heuristic results.

    Args:
        metric_name: The metric being evaluated (e.g. "faithfulness").
        description: What this metric measures (from metric_cls.judge_description).
        inputs: The raw evaluation inputs (output, context, trajectory, etc.).
        local_result: The EvalResult from the local heuristic engine.

    Returns:
        A formatted prompt string ready for the LLM engine.
    """
    return _PROMPT_TEMPLATE.format(
        metric_name=metric_name,
        description=description or f"Evaluate the quality of the output for '{metric_name}'.",
        local_score=local_result.score if local_result.score is not None else "N/A",
        local_reason=local_result.reason or "No reasoning provided.",
        formatted_inputs=_format_inputs(inputs),
    )
