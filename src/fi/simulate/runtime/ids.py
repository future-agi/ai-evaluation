from __future__ import annotations

from uuid import UUID, uuid4, uuid5

_ID_NAMESPACE = UUID("79626429-64cd-4ec6-88e8-97b61bb649f3")


def new_run_id() -> str:
    return f"run_{uuid4().hex}"


def new_plan_id() -> str:
    return f"plan_{uuid4().hex}"


def derive_test_case_id(run_id: str, persona_ref: str, index: int) -> str:
    return _stable_id("case", run_id, persona_ref, str(index))


def derive_event_id(test_case_id: str, source: str, sequence: int) -> str:
    return _stable_id("event", test_case_id, source, str(sequence))


def derive_artifact_id(
    test_case_id: str,
    logical_name: str,
    checksum: str | None = None,
) -> str:
    return _stable_id("artifact", test_case_id, logical_name, checksum or "pending")


def _stable_id(prefix: str, *parts: str) -> str:
    value = "\x1f".join((prefix, *parts))
    return f"{prefix}_{uuid5(_ID_NAMESPACE, value).hex}"
