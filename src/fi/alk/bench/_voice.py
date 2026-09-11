"""Voice modality — deterministic voice-episode verifier (deep contract + simulated).

Voice is the modality that stress-tests the harness: the environment is an active
caller and the verifier is *temporal*, not an exit code. This module scores a
voice **episode transcript** (interleaved caller + agent turns with millisecond
timing) on the dimensions a real voice benchmark cares about:

  * **latency** — the agent answers within the budget after the caller stops;
  * **turn-taking** — no harmful overlap (both speaking at once) outside a
    legitimate barge-in;
  * **barge-in handling** — when the caller interrupts mid-agent-turn, the agent
    yields promptly;
  * **task content** — the agent's words cover the required content.

This is the **simulated / deep-contract** tier: it scores a transcript produced
by a deterministic simulated caller, so it is fully credential-free and
gate-verifiable. The same verifier consumes a transcript captured from a *live*
audio/SIP/WebRTC call + ASR — that live capture (and real WER) is deferred to
owner infra; it plugs in here unchanged by producing the same transcript shape.

A transcript is a list of turns::

    {"speaker": "caller"|"agent", "start_ms": int, "end_ms": int,
     "text": str, "interrupt": bool (optional, caller turns only)}
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

_DEFAULT_MAX_LATENCY_MS = 1200
_BARGE_IN_YIELD_MS = 600  # the agent must stop within this of a barge-in to "yield"
_PASS_FLOOR = 0.75        # each sub-score must meet this for a pass


def _norm_turns(dialogue: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    turns = []
    for t in dialogue:
        turns.append({
            "speaker": str(t.get("speaker")),
            "start_ms": int(t.get("start_ms", 0)),
            "end_ms": int(t.get("end_ms", 0)),
            "text": str(t.get("text") or ""),
            "interrupt": bool(t.get("interrupt", False)),
        })
    return turns


def _latency_score(turns: list[dict[str, Any]], max_latency_ms: int) -> float:
    gaps_ok, gaps = 0, 0
    for i, t in enumerate(turns):
        if t["speaker"] != "agent":
            continue
        prev = next((turns[j] for j in range(i - 1, -1, -1)
                     if turns[j]["speaker"] == "caller"), None)
        if prev is None:
            continue
        gap = t["start_ms"] - prev["end_ms"]
        gaps += 1
        if 0 <= gap <= max_latency_ms:
            gaps_ok += 1
    return gaps_ok / gaps if gaps else 1.0


def _overlap_and_bargein(turns: list[dict[str, Any]]) -> tuple[float, float]:
    """Turn-taking (no harmful overlap) + barge-in handling scores."""

    agent_turns = [t for t in turns if t["speaker"] == "agent"]
    harmful, considered = 0, 0
    bargein_handled, bargein_total = 0, 0
    callers = [t for t in turns if t["speaker"] == "caller"]
    for a in agent_turns:
        considered += 1
        overlapping_callers = [
            c for c in callers
            if c["start_ms"] < a["end_ms"] and c["end_ms"] > a["start_ms"]
        ]
        legit_bargein = False
        for c in overlapping_callers:
            if c["interrupt"]:
                bargein_total += 1
                legit_bargein = True
                # the agent must yield: its turn ends within the window after the
                # interrupt begins.
                if a["end_ms"] - c["start_ms"] <= _BARGE_IN_YIELD_MS:
                    bargein_handled += 1
        if overlapping_callers and not legit_bargein:
            harmful += 1  # both speaking at once with no barge-in to excuse it
    turn_taking = 1.0 - (harmful / considered) if considered else 1.0
    bargein = bargein_handled / bargein_total if bargein_total else 1.0
    return turn_taking, bargein


def _content_score(turns: list[dict[str, Any]], required: Sequence[str]) -> float:
    if not required:
        return 1.0
    agent_text = " ".join(t["text"] for t in turns if t["speaker"] == "agent").lower()
    hit = sum(1 for kw in required if str(kw).lower() in agent_text)
    return hit / len(required)


def score_voice_episode(
    dialogue: Sequence[Mapping[str, Any]],
    *,
    budgets: Mapping[str, Any] | None = None,
    required_content: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Score a voice episode transcript; return ``{"result", "raw"}`` (unified Result).

    The scalar is the mean of the four sub-scores; the verdict (in pass_fail) is a
    pass only if EVERY sub-score meets the floor — a single bad dimension (e.g. the
    agent talks over the caller) fails the episode.
    """

    budgets = budgets or {}
    required_content = required_content or []
    if not dialogue:
        return {
            "result": {"scalar": 0.0, "components": {}, "pass_fail": {"voice": False},
                       "explanation": "empty transcript"},
            "raw": {"modality": "voice"},
        }
    turns = _norm_turns(dialogue)
    max_latency = int(budgets.get("max_latency_ms", _DEFAULT_MAX_LATENCY_MS))
    latency = _latency_score(turns, max_latency)
    turn_taking, bargein = _overlap_and_bargein(turns)
    content = _content_score(turns, required_content)
    sub = {
        "latency": round(latency, 6),
        "turn_taking": round(turn_taking, 6),
        "barge_in": round(bargein, 6),
        "content": round(content, 6),
    }
    scalar = round(sum(sub.values()) / len(sub), 6)
    floors_met = all(v >= _PASS_FLOOR for v in sub.values())
    return {
        "result": {
            "scalar": scalar,
            "components": sub,
            "pass_fail": {"voice": floors_met, **{f"{k}_floor": (v >= _PASS_FLOOR)
                                                  for k, v in sub.items()}},
            "explanation": ("all voice dimensions met the floor" if floors_met
                            else "a voice dimension fell below the floor"),
        },
        "raw": {"modality": "voice", "turns": len(turns), "max_latency_ms": max_latency,
                "floors_met": floors_met},
    }
