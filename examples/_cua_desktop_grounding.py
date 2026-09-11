"""The desktop grounding/step deterministic computation (Phase 9C unit 4).

``grounding_step_accuracy`` is a GENUINELY NEW deterministic computation — it does
NOT exist in ``score_browser_cua_probe_result`` (which scores the browser
post-state). The desktop credential-free rung is screenshot-grounding / step
accuracy ONLY, explicitly NOT full task success (13D-ENV-KINDS §4 computer_use
rung-1 caveat). This computes, deterministically and credential-free, the fraction
of episode steps whose predicted action target matches the ground-truth target —
an element id (exact) OR a coordinate within tolerance.

No VM, no driver, no pyautogui, no playwright, no network, no key.
"""

from __future__ import annotations

from typing import Any, Mapping


def _target_matches(predicted: Mapping[str, Any], ground_truth: Mapping[str, Any], tolerance_px: int) -> bool:
    """A step matches when the predicted target equals the ground-truth target:
    an element id matches exactly; a coordinate matches within ``tolerance_px``
    (Chebyshev distance). Deterministic — no randomness, no model call."""
    if "target_id" in ground_truth:
        return predicted.get("target_id") == ground_truth.get("target_id")
    if "coordinate" in ground_truth:
        pred = predicted.get("coordinate")
        gt = ground_truth.get("coordinate")
        if not (isinstance(pred, (list, tuple)) and isinstance(gt, (list, tuple))):
            return False
        if len(pred) != len(gt):
            return False
        return all(abs(int(a) - int(b)) <= int(tolerance_px) for a, b in zip(pred, gt))
    return False


def grounding_step_accuracy(episode: Mapping[str, Any]) -> float:
    """The deterministic desktop grounding/step anchor: matched_steps / total_steps
    over the committed ``desktop_episode/`` fixture. Byte-identical under repeat
    (no seed dependence — a pure recompute)."""
    steps = list(episode.get("steps") or [])
    tolerance_px = int(episode.get("tolerance_px", 0))
    if not steps:
        return 0.0
    matched = sum(
        1
        for s in steps
        if _target_matches(
            s.get("predicted") or {}, s.get("ground_truth") or {}, tolerance_px
        )
    )
    return round(matched / len(steps), 6)
