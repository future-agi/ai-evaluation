"""
End-to-end integration tests for the revamped evals SDK surface.

Runs against a live backend (defaults to the local `ws2-backend` at
``http://localhost:8003``). Credentials are taken from environment
variables; the test harness auto-logs-in with the test account if they
aren't provided.

Covers:
  * ``Evaluator.evaluate`` response parsing (post-revamp snake_case).
  * ``EvalTemplateManager`` CRUD: list / create / detail / update /
    delete.
  * Versioning: list / create / set-default / restore.
  * Composite eval: create / detail / execute.

Run directly::

    python tests/integration/test_evals_revamp.py
"""

import os
import sys
import time
import traceback
import uuid
from typing import Any, Callable, List, Optional, Tuple

import requests


BASE_URL = os.environ.get("FI_BASE_URL", "http://localhost:8003")
EMAIL = os.environ.get("FI_TEST_EMAIL", "kartik.nvj@futureagi.com")
PASSWORD = os.environ.get("FI_TEST_PASSWORD", "test@123")


def _fetch_api_keys(base_url: str, email: str, password: str) -> Tuple[str, str]:
    """Exchange email+password for an X-Api-Key / X-Secret-Key pair."""
    token_resp = requests.post(
        f"{base_url}/accounts/token/",
        json={"email": email, "password": password},
        timeout=15,
    )
    token_resp.raise_for_status()
    access = token_resp.json()["access"]

    keys_resp = requests.get(
        f"{base_url}/accounts/keys/",
        headers={"Authorization": f"Bearer {access}"},
        timeout=15,
    )
    keys_resp.raise_for_status()
    data = keys_resp.json()["data"]
    return data["api_key"], data["secret_key"]


def _ensure_env() -> None:
    if os.environ.get("FI_API_KEY") and os.environ.get("FI_SECRET_KEY"):
        return
    api_key, secret_key = _fetch_api_keys(BASE_URL, EMAIL, PASSWORD)
    os.environ["FI_API_KEY"] = api_key
    os.environ["FI_SECRET_KEY"] = secret_key
    os.environ["FI_BASE_URL"] = BASE_URL


_ensure_env()

from fi.evals import EvalTemplateManager, Evaluator, Execution  # noqa: E402
from fi.evals.types import EvalResult  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny test runner — no pytest dependency to keep this self-contained.
# ---------------------------------------------------------------------------

_RESULTS: List[Tuple[str, bool, Optional[str]]] = []


def test(name: str):
    def decorator(fn: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            start = time.time()
            try:
                fn()
                elapsed = time.time() - start
                print(f"  PASS  {name}  ({elapsed:.2f}s)")
                _RESULTS.append((name, True, None))
            except Exception as exc:  # noqa: BLE001
                elapsed = time.time() - start
                msg = f"{exc.__class__.__name__}: {exc}"
                print(f"  FAIL  {name}  ({elapsed:.2f}s)\n    {msg}")
                traceback.print_exc()
                _RESULTS.append((name, False, msg))

        run.__name__ = fn.__name__
        return run

    return decorator


def assert_eq(actual: Any, expected: Any, label: str = "") -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(cond: bool, label: str) -> None:
    if not cond:
        raise AssertionError(f"{label} was false")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_evaluator: Optional[Evaluator] = None
_manager: Optional[EvalTemplateManager] = None
_template_id: Optional[str] = None
_second_template_id: Optional[str] = None
_composite_id: Optional[str] = None


def evaluator() -> Evaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = Evaluator()
    return _evaluator


def manager() -> EvalTemplateManager:
    global _manager
    if _manager is None:
        _manager = EvalTemplateManager()
    return _manager


def _random_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@test("Evaluator.evaluate parses revamped snake_case response")
def t_evaluate_tone():
    result = evaluator().evaluate(
        eval_templates="tone",
        inputs={"output": "I absolutely love this product!"},
    )
    assert_true(result is not None, "BatchRunResult returned")
    assert_true(len(result.eval_results) == 1, "exactly one eval result")
    er: EvalResult = result.eval_results[0]
    assert_eq(er.name, "tone", "name")
    assert_true(er.eval_id not in (None, ""), "eval_id populated")
    assert_true(er.output_type not in (None, ""), "output_type populated")
    assert_true(er.output is not None, "output populated")
    assert_true(er.reason is not None and len(er.reason) > 0, "reason populated")


@test("Evaluator.evaluate works for a pass/fail system eval")
def t_evaluate_is_json():
    result = evaluator().evaluate(
        eval_templates="is_json",
        inputs={"text": '{"ok": true, "n": 1}'},
    )
    er = result.eval_results[0]
    assert_true(
        er.output_type in ("Pass/Fail", "pass_fail", "Passed", "Failed"),
        f"output_type reported (got {er.output_type!r})",
    )


@test("EvalTemplateManager.list_templates returns paged results")
def t_list_templates():
    resp = manager().list_templates(page=0, page_size=5)
    assert_true("items" in resp, "items key present")
    assert_true("total" in resp, "total key present")
    assert_true(isinstance(resp["items"], list), "items is list")
    assert_true(len(resp["items"]) <= 5, "respects page_size")


@test("EvalTemplateManager.list_templates system filter")
def t_list_system():
    resp = manager().list_templates(owner_filter="system", page_size=3)
    assert_true(len(resp["items"]) > 0, "system has items")
    assert_true(
        all(item.get("owner") == "system" for item in resp["items"]),
        "all items owned by system",
    )


@test("EvalTemplateManager.create_template + get_template + delete_template")
def t_create_and_delete():
    global _template_id
    name = _random_name("sdk-test-llm")
    created = manager().create_template(
        name=name,
        eval_type="llm",
        instructions="Does the {{output}} sound polite? Answer pass or fail.",
        model="turing_large",
        output_type="pass_fail",
        pass_threshold=0.5,
        description="sdk revamp smoke test",
    )
    assert_true("id" in created, "create returns id")
    assert_eq(created["name"], name, "name matches")
    assert_eq(created["version"], "V1", "version is V1")

    tid = created["id"]
    _template_id = tid

    detail = manager().get_template(tid)
    assert_eq(detail["id"], tid, "detail id")
    assert_eq(detail["name"], name, "detail name")
    assert_eq(detail["eval_type"], "llm", "detail eval_type")
    assert_eq(detail["output_type"], "pass_fail", "detail output_type")
    assert_eq(detail["pass_threshold"], 0.5, "detail pass_threshold")

    # Update description + threshold
    upd = manager().update_template(
        tid,
        description="updated via sdk",
        pass_threshold=0.7,
    )
    assert_eq(upd["id"], tid, "update id")
    assert_true(upd.get("updated"), "update returns updated=True")

    detail2 = manager().get_template(tid)
    assert_eq(detail2["description"], "updated via sdk", "description updated")
    assert_eq(detail2["pass_threshold"], 0.7, "pass_threshold updated")


@test("EvalTemplateManager version CRUD")
def t_versions():
    assert_true(_template_id is not None, "prior test created a template")
    tid = _template_id  # type: ignore[assignment]

    v_list = manager().list_versions(tid)
    assert_eq(v_list["template_id"], tid, "version list template_id")
    assert_true(len(v_list["versions"]) >= 1, "at least V1 exists")

    v1_id = v_list["versions"][-1]["id"]  # oldest = V1 (list is sorted desc)

    v2 = manager().create_version(tid)
    assert_true(v2["version_number"] > 1, f"v2 number > 1 (got {v2['version_number']})")
    v2_id = v2["id"]

    v_list2 = manager().list_versions(tid)
    assert_true(len(v_list2["versions"]) >= 2, "now at least 2 versions")

    set_def = manager().set_default_version(tid, v2_id)
    assert_eq(set_def["id"], v2_id, "set_default returns v2 id")
    assert_true(set_def["is_default"], "is_default=True")

    # Flip default back to v1 via restore (creates a new v3 matching v1).
    restored = manager().restore_version(tid, v1_id)
    assert_true(
        restored["version_number"] > v2["version_number"],
        "restore bumps version_number",
    )


@test("EvalTemplateManager.create_composite + execute")
def t_composite():
    global _composite_id, _second_template_id

    # Need two percentage children for a real composite axis check. Create
    # them fresh so we don't depend on the account's existing inventory.
    a = manager().create_template(
        name=_random_name("sdk-child-a"),
        eval_type="llm",
        instructions="Rate the {{output}} for clarity from 0 to 1.",
        output_type="percentage",
        pass_threshold=0.5,
    )
    b = manager().create_template(
        name=_random_name("sdk-child-b"),
        eval_type="llm",
        instructions="Rate the {{output}} for politeness from 0 to 1.",
        output_type="percentage",
        pass_threshold=0.5,
    )
    _second_template_id = b["id"]

    composite = manager().create_composite(
        name=_random_name("sdk-composite"),
        child_template_ids=[a["id"], b["id"]],
        aggregation_enabled=True,
        aggregation_function="weighted_avg",
        composite_child_axis="percentage",
        child_weights={a["id"]: 1.0, b["id"]: 2.0},
        description="sdk revamp composite smoke test",
    )
    assert_true("id" in composite, "composite created")
    _composite_id = composite["id"]
    assert_eq(composite["template_type"], "composite", "template_type")
    assert_eq(len(composite["children"]), 2, "two children")

    detail = manager().get_composite(_composite_id)
    assert_eq(detail["id"], _composite_id, "composite detail id")
    assert_eq(len(detail["children"]), 2, "two children in detail")

    run = manager().execute_composite(
        _composite_id,
        mapping={"output": "Hello, I hope you're having a wonderful day!"},
    )
    assert_eq(run["composite_id"], _composite_id, "execute composite id")
    assert_eq(run["total_children"], 2, "total_children=2")
    assert_true(
        run["completed_children"] + run["failed_children"] == 2,
        "every child accounted for",
    )
    assert_true(isinstance(run.get("children"), list), "children list returned")


@test("Evaluator.submit returns execution handle and wait() completes")
def t_submit_eval_wait():
    handle = evaluator().submit(
        "tone",
        {"output": "I absolutely love this product!"},
    )
    assert_true(isinstance(handle, Execution), "Execution returned")
    assert_eq(handle.kind, "eval", "kind=eval")
    assert_true(handle.id not in (None, ""), "execution id populated")
    assert_true(
        handle.status in ("pending", "processing"),
        f"initial status is non-terminal (got {handle.status!r})",
    )

    handle.wait(timeout=120, poll_interval=2)
    assert_eq(handle.status, "completed", "terminal status")
    assert_true(handle.result is not None, "result populated")
    assert_eq(handle.result.name, "tone", "result name")
    assert_true(handle.result.output is not None, "result output")


@test("Evaluator.get_execution re-attaches to a submitted handle by id")
def t_get_execution_by_id():
    handle = evaluator().submit("tone", {"output": "Have a great day!"})
    exec_id = handle.id
    refetched = evaluator().get_execution(exec_id)
    assert_eq(refetched.id, exec_id, "id matches")
    assert_eq(refetched.kind, "eval", "kind=eval")
    refetched.wait(timeout=120, poll_interval=2)
    assert_eq(refetched.status, "completed", "completed via refetched handle")
    assert_true(refetched.result is not None, "result populated")


@test("Evaluator.submit with error_localizer=True surfaces analysis")
def t_submit_with_error_localizer():
    handle = evaluator().submit(
        "toxicity",
        {"output": "You are a worthless idiot, I hate you!"},
        error_localizer=True,
    )
    handle.wait(timeout=180, poll_interval=3)
    assert_eq(handle.status, "completed", "completed")
    assert_true(
        handle.result is not None and handle.result.error_localizer_enabled is True,
        "error_localizer_enabled=True",
    )


@test("EvalTemplateManager.submit_composite returns handle and wait() completes")
def t_submit_composite_wait():
    assert_true(_composite_id is not None, "prior test created composite")
    handle = manager().submit_composite(
        _composite_id,  # type: ignore[arg-type]
        mapping={"output": "Thanks, have a lovely day!"},
    )
    assert_true(isinstance(handle, Execution), "Execution returned")
    assert_eq(handle.kind, "composite", "kind=composite")
    assert_true(handle.id not in (None, ""), "local execution id populated")
    assert_true(
        handle.status in ("pending", "processing"),
        f"initial status non-terminal (got {handle.status!r})",
    )

    handle.wait(timeout=180, poll_interval=2)
    assert_eq(handle.status, "completed", "terminal status")
    assert_true(isinstance(handle.result, dict), "result is dict")
    assert_eq(handle.result.get("composite_id"), _composite_id, "composite_id matches")
    assert_eq(handle.result.get("total_children"), 2, "total_children=2")


@test("Cleanup: delete created templates")
def t_cleanup():
    ids: List[str] = []
    if _template_id:
        ids.append(_template_id)
    if _second_template_id:
        ids.append(_second_template_id)
    if _composite_id:
        ids.append(_composite_id)

    for tid in ids:
        try:
            manager().delete_template(tid)
        except Exception as exc:  # noqa: BLE001
            print(f"    warning: failed to delete {tid}: {exc}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"Using backend: {BASE_URL}")
    print(f"Test user: {EMAIL}")
    print()

    tests = [
        t_evaluate_tone,
        t_evaluate_is_json,
        t_list_templates,
        t_list_system,
        t_create_and_delete,
        t_versions,
        t_composite,
        t_submit_eval_wait,
        t_get_execution_by_id,
        t_submit_with_error_localizer,
        t_submit_composite_wait,
        t_cleanup,
    ]
    for fn in tests:
        print(f"Running: {fn.__name__}")
        fn()
    print()

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    failed = sum(1 for _, ok, _ in _RESULTS if not ok)
    print(f"Summary: {passed} passed, {failed} failed, {len(_RESULTS)} total")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
