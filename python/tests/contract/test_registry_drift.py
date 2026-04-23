"""Detect SDK ⇄ api drift.

Goals:
  - Every `EvalTemplate` subclass in the SDK names an eval that actually
    exists in the api registry (or is explicitly marked DEPRECATED).
  - Every eval in the api registry with `eval_type_id=='AgentEvaluator'`
    has some way to be invoked — either a class or the string path (which
    works thanks to TuringEngine's string fallback).

If these fail, the SDK and api have drifted. Fix the SDK before merging.
"""
import inspect

import pytest

from fi.evals import templates as tmpl_mod
from fi.evals.templates import EvalTemplate


def _sdk_template_classes() -> dict:
    """{eval_name: class} for every concrete EvalTemplate subclass."""
    out = {}
    for _, obj in inspect.getmembers(tmpl_mod, inspect.isclass):
        if obj is EvalTemplate or not issubclass(obj, EvalTemplate):
            continue
        name = getattr(obj, "eval_name", None)
        if name:
            out[name] = obj
    return out


def _is_deprecated(cls: type) -> bool:
    """Docstring marker — classes we know the api has removed."""
    return "DEPRECATED" in (cls.__doc__ or "")


def test_every_non_deprecated_sdk_class_exists_on_api(live_registry):
    """SDK says `Toxicity` exists; api must agree."""
    backend_names = set(live_registry.keys())
    sdk = _sdk_template_classes()

    missing = []
    for name, cls in sdk.items():
        if _is_deprecated(cls):
            continue
        if name not in backend_names:
            missing.append(f"{cls.__name__}(eval_name={name!r})")

    assert not missing, (
        "SDK template classes reference evals the api doesn't ship. "
        "Either mark them DEPRECATED in the docstring, delete them, or "
        "get the api team to add them:\n  - " + "\n  - ".join(sorted(missing))
    )


def test_every_deprecated_class_has_justification(live_registry):
    """If we call a class DEPRECATED, the eval really shouldn't be on the api."""
    backend_names = set(live_registry.keys())
    sdk = _sdk_template_classes()

    spurious = [
        f"{cls.__name__}(eval_name={name!r})"
        for name, cls in sdk.items()
        if _is_deprecated(cls) and name in backend_names
    ]
    assert not spurious, (
        "These classes are marked DEPRECATED but the eval IS on the api. "
        "Remove the DEPRECATED tag:\n  - " + "\n  - ".join(sorted(spurious))
    )


def test_required_keys_stable_for_known_evals(live_registry):
    """Spot-check a few evals where key rename is catastrophic.

    These are evals we've been burned on. If the api renames a required
    key, fail loudly so the SDK alias map gets updated.
    """
    expectations = {
        "toxicity": {"output"},
        "prompt_injection": {"input"},
        "factual_accuracy": {"input", "output", "context"},
        "groundedness": {"input", "output", "context"},
        "conversation_coherence": {"conversation"},
        "fuzzy_match": {"expected", "output"},
        "bleu_score": {"reference", "hypothesis"},
    }

    drift = []
    for name, expected in expectations.items():
        info = live_registry.get(name)
        if info is None:
            continue  # covered by the existence test above
        actual = set(info.get("config", {}).get("required_keys", []) or [])
        if actual != expected:
            drift.append(f"{name}: expected {sorted(expected)}, got {sorted(actual)}")

    assert not drift, (
        "Required-key drift detected. Update the alias map in "
        "fi/evals/core/cloud_registry.py or fix the api:\n  - "
        + "\n  - ".join(drift)
    )
