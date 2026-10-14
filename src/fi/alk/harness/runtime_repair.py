"""Evidence-backed repair of the generated runtime plan.

This module may revise only the runtime metadata in the generated contract.  The submitted
repository is mounted read-only and remains the authority; agent behavior, source files,
credentials, egress policy, tools and scenarios are outside this repair boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .backends import FILE_TOOLS, SessionSpec, tool, tool_server
from .contract import AgentContract, Runtime
from .diagnostics import HarnessDiagnostic
from .session import Stage


_SAFE_RUNTIME_LAUNCHERS = {
    "python",
    "python3",
    "python3.10",
    "python3.11",
    "python3.12",
    "python3.13",
    "uv",
    "poetry",
    "node",
    "npm",
    "npx",
    "pnpm",
    "yarn",
}
_SHELL_LAUNCHERS = {"sh", "bash", "zsh", "dash", "fish", "cmd", "powershell"}


def _contract_hash(contract: AgentContract) -> str:
    body = contract.model_dump_json(exclude_none=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(body).hexdigest()


class RuntimePlanPatch(BaseModel):
    """A complete replacement for generated runtime metadata, tied to source evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_contract_hash: str
    runtime: Runtime
    evidence_paths: tuple[str, ...] = Field(min_length=1, max_length=32)
    diagnostic_codes: tuple[str, ...] = Field(min_length=1, max_length=16)
    summary: str = Field(min_length=1, max_length=1000)

    @field_validator("evidence_paths")
    @classmethod
    def _portable_evidence(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = []
        for value in values:
            path = Path(str(value).strip())
            if not str(path) or path.is_absolute() or ".." in path.parts:
                raise ValueError("runtime_repair_evidence_path_must_be_relative")
            normalized.append(path.as_posix())
        return tuple(dict.fromkeys(normalized))


class _RuntimeSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime: Runtime
    evidence_paths: tuple[str, ...] = Field(min_length=1, max_length=32)
    summary: str = Field(min_length=1, max_length=1000)


def validate_runtime_plan_patch(
    source_root: Path,
    contract: AgentContract,
    patch: RuntimePlanPatch,
    *,
    allowed_diagnostic_codes: set[str],
) -> AgentContract:
    """Validate a proposed plan without executing or changing submitted source."""

    root = source_root.resolve()
    if patch.base_contract_hash != _contract_hash(contract):
        raise ValueError("runtime_repair_base_contract_hash_mismatch")
    if not set(patch.diagnostic_codes) or not set(patch.diagnostic_codes).issubset(
        allowed_diagnostic_codes
    ):
        raise ValueError("runtime_repair_diagnostic_code_not_allowed")
    evidence_files: list[Path] = []
    for relative in patch.evidence_paths:
        evidence = (root / relative).resolve()
        try:
            evidence.relative_to(root)
        except ValueError as exc:
            raise ValueError("runtime_repair_evidence_escapes_repository") from exc
        if not evidence.exists():
            raise ValueError(f"runtime_repair_evidence_missing: {relative}")
        if evidence.is_file():
            evidence_files.append(evidence)

    runtime = patch.runtime
    workdir = (root / runtime.workdir).resolve() if runtime.workdir else root
    try:
        workdir.relative_to(root)
    except ValueError as exc:
        raise ValueError("runtime_repair_workdir_escapes_repository") from exc
    if not workdir.is_dir():
        raise ValueError("runtime_repair_workdir_missing")
    if runtime.command:
        launcher = Path(runtime.command[0]).name.casefold()
        if launcher in _SHELL_LAUNCHERS or any(
            token in {"-c", "--eval", "-e"} for token in runtime.command[1:]
        ):
            raise ValueError("runtime_repair_inline_or_shell_command_forbidden")
        if launcher not in _SAFE_RUNTIME_LAUNCHERS:
            declared = "\n".join(
                path.read_text(encoding="utf-8", errors="replace")[:100_000]
                for path in evidence_files
            )
            if runtime.command[0] not in declared:
                raise ValueError("runtime_repair_launcher_not_source_evidenced")
        for token in runtime.command[1:]:
            if Path(token).suffix.casefold() not in {".py", ".js", ".mjs", ".cjs"}:
                continue
            entrypoint = (workdir / token).resolve()
            try:
                entrypoint.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    "runtime_repair_entrypoint_escapes_repository"
                ) from exc
            if not entrypoint.is_file():
                raise ValueError(f"runtime_repair_entrypoint_missing: {token}")
    for configured, label in (
        (runtime.compose_file, "compose_file"),
        (runtime.dockerfile, "dockerfile"),
    ):
        if configured and (
            not (root / configured).resolve().is_relative_to(root)
            or not (root / configured).is_file()
        ):
            raise ValueError(f"runtime_repair_{label}_missing")

    from .bundle_author_v2 import _declared_runtime_environment
    from .runtime_selection import select_runtime

    select_runtime(
        root, runtime, public_environment=_declared_runtime_environment(root)
    )
    candidate = contract.model_copy(update={"runtime": runtime})
    # Revalidate the whole contract so nested updates cannot bypass canonical validators.
    return AgentContract.model_validate(candidate.model_dump(mode="python"))


def apply_runtime_plan_patch(
    source_root: Path,
    contract_path: Path,
    patch: RuntimePlanPatch,
    *,
    allowed_diagnostic_codes: set[str],
) -> AgentContract:
    """Atomically apply only a validated runtime metadata replacement."""

    contract = AgentContract.model_validate_json(
        contract_path.read_text(encoding="utf-8")
    )
    repaired = validate_runtime_plan_patch(
        source_root,
        contract,
        patch,
        allowed_diagnostic_codes=allowed_diagnostic_codes,
    )
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".contract-runtime-repair-", suffix=".json", dir=contract_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(repaired.model_dump_json(indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, contract_path)
    finally:
        temporary.unlink(missing_ok=True)
    return repaired


async def request_runtime_plan_patch(
    source_root: Path,
    contract: AgentContract,
    diagnostics: tuple[HarnessDiagnostic, ...],
    *,
    guidance: str = "",
) -> RuntimePlanPatch:
    """Inspect an unfamiliar repository and submit one constrained runtime-plan revision."""

    allowed = {item.code for item in diagnostics}
    captured: list[RuntimePlanPatch] = []

    @tool(
        "submit_runtime_plan_patch",
        "Submit a complete evidence-backed replacement for generated runtime metadata. "
        "This cannot edit source, agent behavior, tools, scenarios, credentials, or policy.",
        _RuntimeSubmission.model_json_schema(),
    )
    async def submit(arguments: dict[str, object]) -> dict[str, object]:
        try:
            submission = _RuntimeSubmission.model_validate(arguments)
            patch = RuntimePlanPatch(
                base_contract_hash=_contract_hash(contract),
                runtime=submission.runtime,
                evidence_paths=submission.evidence_paths,
                diagnostic_codes=tuple(sorted(allowed)),
                summary=submission.summary,
            )
            validate_runtime_plan_patch(
                source_root,
                contract,
                patch,
                allowed_diagnostic_codes=allowed,
            )
        except Exception as exc:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"patch rejected: {type(exc).__name__}: {exc}",
                    }
                ],
                "is_error": True,
            }
        captured[:] = [patch]
        return {"content": [{"type": "text", "text": "runtime plan patch accepted"}]}

    diagnostics_json = [
        item.model_dump(mode="json", exclude={"redacted_message"})
        | {"message": item.redacted_message}
        for item in diagnostics
    ]
    system_prompt = """You repair how an arbitrary submitted agent repository is built and
started. First inspect repository evidence with Read, Glob, and Grep. Prefer declared manifests,
scripts, container files, framework configuration, and real server/entrypoint code. Preserve the
existing deterministic plan when it is already supported; change only fields proven wrong by the
diagnostics. Never edit source, invent agent behavior or services, add credentials, widen egress,
change tools/scenarios, or hide a failure. Your only write capability is
submit_runtime_plan_patch. The replacement must use paths and commands that exist in the submitted
repository. Call the tool once the smallest complete correction is known, then stop."""
    briefing = (
        "Repair the generated runtime plan for this repository.\n\nDIAGNOSTICS\n"
        + json.dumps(diagnostics_json, sort_keys=True, indent=2)[:20000]
        + "\n\nCURRENT CONTRACT\n"
        + contract.model_dump_json(indent=2)[:36000]
        + ("\n\nREQUIRED JOB INTENT\n" + guidance if guidance else "")
    )
    stage = Stage(
        SessionSpec(
            system_prompt=system_prompt,
            servers={"runtime_repair": tool_server("runtime_repair", tools=[submit])},
            builtins=FILE_TOOLS,
            cwd=str(source_root.resolve()),
            max_turns=50,
            gated=True,
            thinking=True,
        ),
        name="repair-runtime-plan",
    )
    async with stage:
        await stage.say(briefing)
        if not captured:
            await stage.say(
                "No valid runtime plan was submitted. Inspect the remaining evidence and call "
                "submit_runtime_plan_patch now."
            )
    if not captured:
        raise RuntimeError("typed_runtime_plan_patch_not_submitted")
    return captured[0]


__all__ = [
    "RuntimePlanPatch",
    "apply_runtime_plan_patch",
    "request_runtime_plan_patch",
    "validate_runtime_plan_patch",
]
