"""Judge-reliability evaluation over scripted judge outputs.

Deterministic, offline, credential-free. A scripted rubric judge (a pure
Python scoring function) scores fixed sample outputs, then the same outputs
are perturbed along three axes — formatting, verbosity, and hardcoded
paraphrase variants (no LLM calls) — and the score deltas across
perturbations are measured and asserted against a tolerance. The artifact is
an `agent-learning.eval.v1` payload whose summary carries the agreement
metrics per axis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import evals


JUDGE_PASS_THRESHOLD = 0.7
AGREEMENT_DELTA_TOLERANCE = 0.15

SAMPLES: list[dict[str, Any]] = [
    {
        "id": "refund-window",
        "required_terms": ["refund", "30 days", "receipt"],
        "output": (
            "You can request a refund within 30 days of purchase. Keep your "
            "receipt and submit the request from the orders page."
        ),
        "paraphrase": (
            "A refund stays available for 30 days after you buy. Hold on to "
            "the receipt and file the request through the orders page."
        ),
    },
    {
        "id": "shipping-status",
        "required_terms": ["tracking number", "48 hours", "carrier"],
        "output": (
            "Your tracking number is issued within 48 hours of dispatch, and "
            "the carrier updates the status once the parcel is scanned."
        ),
        "paraphrase": (
            "Within 48 hours of dispatch we issue the tracking number; the "
            "carrier refreshes the status after the first scan."
        ),
    },
    {
        "id": "password-reset",
        "required_terms": ["reset link", "15 minutes", "spam folder"],
        "output": (
            "Use the reset link emailed to you; it expires after 15 minutes. "
            "If it is missing, check your spam folder before retrying."
        ),
        "paraphrase": (
            "Open the reset link from the email within 15 minutes, since it "
            "expires. Look in the spam folder if nothing arrived."
        ),
    },
]

VERBOSITY_FILLER = (
    " To summarize the points above in additional detail, the team reviewed "
    "the request, confirmed the account context, checked the relevant "
    "internal notes, and validated that the answer matches current "
    "documentation before sending this response."
)


def judge_score(text: str, required_terms: list[str]) -> float:
    """Scripted judge: rubric-term coverage with a verbosity penalty."""

    lowered = " ".join(text.lower().split())
    coverage = sum(1 for term in required_terms if term in lowered) / len(
        required_terms
    )
    penalty = 0.05 if len(lowered.split()) > 45 else 0.0
    return round(max(0.0, coverage - penalty), 4)


def perturb_formatting(text: str) -> str:
    sentences = [chunk.strip() for chunk in text.split(". ") if chunk.strip()]
    return "ANSWER:\n" + "\n".join(f"- {s.rstrip('.')}." for s in sentences)


def perturb_verbosity(text: str) -> str:
    return text + VERBOSITY_FILLER


def perturb_paraphrase(sample: dict[str, Any]) -> str:
    return str(sample["paraphrase"])


def measure_agreement() -> dict[str, Any]:
    axes: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    perturbations = {
        "formatting": lambda sample: perturb_formatting(sample["output"]),
        "verbosity": lambda sample: perturb_verbosity(sample["output"]),
        "paraphrase": perturb_paraphrase,
    }
    for axis, perturb in perturbations.items():
        deltas: list[float] = []
        verdict_flips = 0
        for sample in SAMPLES:
            terms = list(sample["required_terms"])
            base = judge_score(str(sample["output"]), terms)
            variant = judge_score(perturb(sample), terms)
            delta = round(abs(variant - base), 4)
            flipped = (base >= JUDGE_PASS_THRESHOLD) != (
                variant >= JUDGE_PASS_THRESHOLD
            )
            verdict_flips += int(flipped)
            deltas.append(delta)
            rows.append(
                {
                    "id": str(sample["id"]),
                    "axis": axis,
                    "base_score": base,
                    "perturbed_score": variant,
                    "score_delta": delta,
                    "verdict_flipped": flipped,
                }
            )
        axes[axis] = {
            "mean_score_delta": round(sum(deltas) / len(deltas), 4),
            "max_score_delta": max(deltas),
            "verdict_agreement": round(1.0 - verdict_flips / len(SAMPLES), 4),
        }
    return {"axes": axes, "results": rows}


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    measured = measure_agreement()
    axes = measured["axes"]
    max_delta = max(axis["max_score_delta"] for axis in axes.values())
    mean_delta = round(
        sum(axis["mean_score_delta"] for axis in axes.values()) / len(axes), 4
    )
    verdict_agreement = min(axis["verdict_agreement"] for axis in axes.values())
    within_tolerance = (
        max_delta <= AGREEMENT_DELTA_TOLERANCE and verdict_agreement == 1.0
    )
    assert within_tolerance, (
        f"judge agreement drift exceeded tolerance: max_delta={max_delta}, "
        f"verdict_agreement={verdict_agreement}"
    )

    result: dict[str, Any] = {
        "kind": evals.AGENT_LEARNING_EVAL_KIND,
        "schema_version": evals.AGENT_LEARNING_EVAL_KIND,
        "name": "sdk-judge-reliability-evaluation",
        "status": "passed" if within_tolerance else "failed",
        "exit_code": 0 if within_tolerance else 1,
        "summary": {
            "sample_count": len(SAMPLES),
            "perturbation_axes": sorted(axes),
            "judge_pass_threshold": JUDGE_PASS_THRESHOLD,
            "delta_tolerance": AGREEMENT_DELTA_TOLERANCE,
            "max_score_delta": max_delta,
            "mean_score_delta": mean_delta,
            "verdict_agreement": verdict_agreement,
            "axes": axes,
        },
        "results": measured["results"],
    }
    if output_path is not None:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return result


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    payload = run(destination)
    if destination is None:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
