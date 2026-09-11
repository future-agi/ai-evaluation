from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, optimize, simulate


REQUIRED_ENV = "AGENT_LEARNING_SDK_CAPABILITY_FREEZE_EXAMPLE_KEY"
FROZEN_PROFILE_FIXTURE = (
    Path(__file__).resolve().parent
    / "frozen_profiles"
    / "frozen_capability_profile.json"
)
FREEZE_FRAMEWORKS = ["langgraph", "livekit"]
FREEZE_SETTING = {
    "engine": "local_text",
    "driver": "deterministic_scripted",
    "eval_budget": 8,
    "required_env": [REQUIRED_ENV],
    "target_kind": "whole_agent",
}
FREEZE_METRIC_FLOORS = {
    "task_completion": 0.9,
    "adapter_contract_coverage": 1.0,
}
FREEZE_SECURITY_ROWS = [
    {
        "framework": "all",
        "capability": "stored_injection_resilience",
        "metric": "redteam_pass_rate",
        "floor": 1.0,
        "source": "redteam.stored_injection_readiness",
    }
]
FROZEN_AT = "2026-06-11T00:00:00Z"


def build_frozen_profile() -> dict[str, Any]:
    """Freeze the capability-profile bundle into the evidence contract."""

    profiles = simulate.framework_adapter_capability_profiles(
        frameworks=list(FREEZE_FRAMEWORKS),
    )
    return optimize.freeze_capability_profile(
        profiles,
        setting=FREEZE_SETTING,
        metric_floors=FREEZE_METRIC_FLOORS,
        security_rows=FREEZE_SECURITY_ROWS,
        frozen_at=FROZEN_AT,
        source_manifest_ref="examples/sdk_capability_freeze_regression.py",
    )


def _candidate(
    *,
    metric_averages: dict[str, float],
    setting: dict[str, Any],
    patch: dict[str, Any] | None = None,
    searched_metric_gain: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A deterministic agent-learning.optimization.v1-shaped candidate result."""

    payload: dict[str, Any] = {
        "kind": "agent-learning.optimization.v1",
        "status": "passed",
        "setting": dict(setting),
        "summary": {"metric_averages": dict(metric_averages)},
        "optimization": {"history": []},
    }
    if patch is not None:
        payload["patch"] = dict(patch)
    if searched_metric_gain is not None:
        payload["searched_metric_gain"] = dict(searched_metric_gain)
    return payload


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    frozen = build_frozen_profile()
    committed = json.loads(FROZEN_PROFILE_FIXTURE.read_text(encoding="utf-8"))
    fixture_match = committed == frozen

    promotion = optimize.attach_frozen_profile(
        {
            "kind": "agent-learning.regression-promotion.v1",
            "name": "sdk-capability-freeze-regression",
            "summary": {},
        },
        frozen,
    )

    closing_metrics = {
        "task_completion": 0.95,
        "adapter_contract_coverage": 1.0,
        "redteam_pass_rate": 1.0,
    }

    # 1. Compliant candidate: every frozen row re-closes under the same
    # setting digest -> promotable.
    compliant = optimize.replay_frozen_profile(
        _candidate(metric_averages=closing_metrics, setting=FREEZE_SETTING),
        frozen,
    )

    # 2. The PRD §4.1 negative fixture: the candidate IMPROVES its searched
    # metric while breaking one frozen row -> vetoed (badhita), regardless of
    # the win.
    improving_but_breaking = optimize.replay_frozen_profile(
        _candidate(
            metric_averages={
                "task_completion": 0.99,
                "adapter_contract_coverage": 0.4,
                "redteam_pass_rate": 1.0,
            },
            setting=FREEZE_SETTING,
            searched_metric_gain={
                "metric": "task_completion",
                "baseline": 0.91,
                "candidate": 0.99,
            },
        ),
        frozen,
    )

    # 3. Out-of-setting win: same scores, different declared setting digest ->
    # rows are non-admissible; the win does not count (orderings invert
    # across settings).
    out_of_setting = optimize.replay_frozen_profile(
        _candidate(
            metric_averages=closing_metrics,
            setting={**FREEZE_SETTING, "eval_budget": 64},
        ),
        frozen,
    )

    # 4. Security rows are non-tradable: a candidate patch touching
    # context-memory paths with the security row not re-passed at floor is
    # vetoed regardless of score.
    security_trade = optimize.replay_frozen_profile(
        _candidate(
            metric_averages={**closing_metrics, "redteam_pass_rate": 0.5},
            setting=FREEZE_SETTING,
            patch={"memory.retrieval.depth": 2},
            searched_metric_gain={
                "metric": "task_completion",
                "baseline": 0.91,
                "candidate": 0.95,
            },
        ),
        frozen,
    )

    # 5. Tampered row_id: mutating a frozen row without recomputing its
    # content address is detected (asiddha — the cited row is not the row).
    tampered = json.loads(json.dumps(frozen))
    tampered["rows"][0]["floor"] = 0.0
    tampered_row = optimize.replay_frozen_profile(
        _candidate(metric_averages=closing_metrics, setting=FREEZE_SETTING),
        tampered,
    )

    # The veto is recorded in governance as a steward nirnaya entry: the
    # improvement is rejected over the frozen-row regression, citing row_ids.
    nirnaya = {
        "decision": "reject_candidate",
        "round": 1,
        "selected_candidate_id": None,
        "rejected_alternatives": [
            {
                "candidate_id": "candidate_improving_but_breaking",
                "hetvabhasa_class": improving_but_breaking["hetvabhasa_class"],
                "vetoed_row_ids": [
                    row["row_id"]
                    for row in improving_but_breaking["vetoed_rows"]
                ],
            }
        ],
        "replay_verdict": "frozen_row_regression",
        "frozen_rows_closed": improving_but_breaking["closed_row_count"],
        "frozen_profile_ref": frozen["contract_digest"],
    }

    rows = list(frozen["rows"])
    checks = {
        "rows_content_addressed": fixture_match
        and all(row.get("integrity_ok") is not False for row in compliant["rows"])
        and all(str(row.get("row_id", "")).startswith("row_") for row in rows),
        "improving_candidate_with_broken_row_vetoed": (
            improving_but_breaking["veto"] is True
            and improving_but_breaking["hetvabhasa_class"] == "badhita"
        ),
        "veto_recorded_in_governance": bool(
            nirnaya["rejected_alternatives"]
            and nirnaya["rejected_alternatives"][0]["hetvabhasa_class"]
            == "badhita"
            and nirnaya["rejected_alternatives"][0]["vetoed_row_ids"]
        ),
        "out_of_setting_win_non_admissible": (
            bool(out_of_setting["non_admissible_wins"])
            and len(out_of_setting["non_admissible_wins"])
            == len(rows)
        ),
        "security_row_non_tradable": (
            security_trade["veto"] is True
            and security_trade["security_veto"] is True
            and security_trade["touches_context_memory_paths"] is True
        ),
    }

    result: dict[str, Any] = {
        "kind": "agent-learning.capability-profile-freeze-regression.v1",
        "schema_version": "agent-learning.cli.v1",
        "status": "passed" if all(checks.values()) else "failed",
        "required_env": [REQUIRED_ENV],
        "frozen": frozen,
        "fixture": {
            "path": "examples/frozen_profiles/frozen_capability_profile.json",
            "match": fixture_match,
            "contract_digest": frozen["contract_digest"],
            "setting_digest": frozen["setting_digest"],
        },
        "promotion": {
            "kind": promotion.get("kind"),
            "frozen_capability_profile_digest": promotion["summary"][
                "frozen_capability_profile_digest"
            ],
            "frozen_capability_profile_row_count": promotion["summary"][
                "frozen_capability_profile_row_count"
            ],
        },
        "replays": {
            "compliant": compliant,
            "improving_but_breaking": improving_but_breaking,
            "out_of_setting": out_of_setting,
            "security_trade": security_trade,
            "tampered_row": tampered_row,
        },
        "governance": {"nirnaya": [nirnaya]},
        "checks": checks,
        "summary": {
            "row_count": len(rows),
            "security_row_count": sum(1 for row in rows if row.get("security")),
            "check_count": len(checks),
            "passed_check_count": sum(1 for value in checks.values() if value),
            "failed_checks": sorted(
                name for name, value in checks.items() if not value
            ),
        },
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
