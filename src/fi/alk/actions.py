from __future__ import annotations

import contextlib
import copy
import importlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from ._schema import AGENT_LEARNING_CLI_SCHEMA_VERSION, normalize_public_payload

AGENT_LEARNING_ACTIONS_KIND = "agent-learning.actions.v1"
AGENT_LEARNING_ACTION_RUN_KIND = "agent-learning.action-run.v1"
_MISSING = object()


def load_artifact_file(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path).expanduser().resolve()
    loaded = _load_json_or_yaml(artifact_path)
    if not isinstance(loaded, Mapping):
        raise ValueError("action artifact root must be an object")
    return dict(loaded)


def extract_actions(
    artifact: Mapping[str, Any],
    *,
    action_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return embedded executable actions from an Agent Learning artifact/report."""

    normalized = normalize_public_payload(artifact)
    if not isinstance(normalized, Mapping):
        return []
    actions: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for path, action in _walk_actions(normalized):
        if action_id is not None and str(action.get("id") or "") != action_id:
            continue
        record = copy.deepcopy(dict(action))
        record["path"] = path
        record["source_card_path"] = _source_card_path(path)
        key = (
            str(record.get("id") or ""),
            str(record.get("kind") or ""),
            str(record.get("command") or ""),
            str(record.get("artifact_ref") or ""),
            str(record.get("path") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        actions.append(record)
    return actions


def get_action(
    artifact: Mapping[str, Any],
    action_id: str,
    *,
    source_path: str | Path = ".",
) -> Optional[dict[str, Any]]:
    actions = action_catalog(
        artifact,
        source_path=source_path,
        action_id=action_id,
    )["actions"]
    return actions[0] if actions else None


def run_action(
    artifact: Mapping[str, Any],
    action_id: str,
    *,
    source_path: str | Path = ".",
    inputs: Optional[Mapping[str, Any]] = None,
    cwd: str | Path | None = None,
    dry_run: bool = False,
    name: Optional[str] = None,
    artifact_output_path: str | Path | None = None,
) -> dict[str, Any]:
    action = get_action(artifact, action_id, source_path=source_path)
    if action is None:
        raise ValueError(f"action not found: {action_id}")
    action_kind = str(action.get("kind") or "cli")
    if action_kind == "download":
        return _run_download_action(
            artifact,
            action_id,
            action,
            source_path=source_path,
            inputs=inputs or {},
            cwd=cwd,
            dry_run=dry_run,
            name=name,
            artifact_output_path=artifact_output_path,
        )
    if action_kind != "cli":
        raise ValueError(f"unsupported action kind: {action_kind}")

    command_args = _resolved_command_args(action, inputs or {})
    if not command_args:
        raise ValueError(f"action {action_id!r} does not include command_args")
    command_name = command_args[0]
    if command_name != "agent-learn":
        raise ValueError(
            f"unsupported action command: {command_name}; use agent-learn"
        )
    if len(command_args) < 2:
        raise ValueError(f"action {action_id!r} is missing a subcommand")
    subcommand = command_args[1]
    if subcommand in {"action-run", "run-action"}:
        raise ValueError("action-run cannot recursively execute action-run")

    run_cwd = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd()
    command_args = _absolutize_output_args(command_args, run_cwd)
    output_records = _command_output_records(command_args, run_cwd)
    exit_code = 0
    logs = {"stdout": "", "stderr": "", "stdout_bytes": 0, "stderr_bytes": 0}
    if not dry_run:
        dispatch = _dispatch_action_command(command_args, cwd=run_cwd)
        exit_code = int(dispatch["exit_code"])
        logs = {
            "stdout": str(dispatch.get("stdout") or ""),
            "stderr": str(dispatch.get("stderr") or ""),
            "stdout_bytes": int(dispatch.get("stdout_bytes") or 0),
            "stderr_bytes": int(dispatch.get("stderr_bytes") or 0),
        }
        output_records = _command_output_records(command_args, run_cwd)
    status = "passed" if exit_code == 0 else "failed"
    outputs_written_count = sum(
        1 for item in output_records if item.get("exists") is True
    )
    output_count = len(output_records)
    output_completion_rate = (
        round(outputs_written_count / output_count, 4)
        if output_count
        else 1.0
    )
    payload = {
        "schema_version": AGENT_LEARNING_CLI_SCHEMA_VERSION,
        "kind": AGENT_LEARNING_ACTION_RUN_KIND,
        "name": str(name or f"{action_id}-action-run"),
        "status": status,
        "exit_code": exit_code,
        "source_path": str(source_path),
        "cwd": str(run_cwd),
        "dry_run": bool(dry_run),
        "action": action,
        "command": " ".join(_shell_token(arg) for arg in command_args),
        "command_args": command_args,
        "logs": logs,
        "outputs": output_records,
        "outputs_written": [
            str(item["path"])
            for item in output_records
            if item.get("exists") is True
        ],
        "summary": {
            "action_id": str(action.get("id") or action_id),
            "action_label": action.get("label"),
            "action_kind": action_kind,
            "source_card_path": action.get("source_card_path"),
            "requires_input": bool(action.get("inputs")),
            "command_exit_code": exit_code,
            "stdout_bytes": logs["stdout_bytes"],
            "stderr_bytes": logs["stderr_bytes"],
            "output_count": output_count,
            "outputs_written_count": outputs_written_count,
            "output_completion_rate": output_completion_rate,
        },
    }
    return payload


def _run_download_action(
    artifact: Mapping[str, Any],
    action_id: str,
    action: Mapping[str, Any],
    *,
    source_path: str | Path,
    inputs: Mapping[str, Any],
    cwd: str | Path | None,
    dry_run: bool,
    name: Optional[str],
    artifact_output_path: str | Path | None,
) -> dict[str, Any]:
    artifact_ref = str(action.get("artifact_ref") or "")
    if not artifact_ref:
        raise ValueError(f"download action {action_id!r} is missing artifact_ref")

    value = _resolve_artifact_ref(
        artifact,
        artifact_ref,
        source_path=source_path,
    )
    if value is _MISSING:
        raise ValueError(
            f"download action {action_id!r} artifact_ref not found: {artifact_ref}"
        )

    run_cwd = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd()
    output_path = _download_output_path(
        action,
        inputs,
        run_cwd,
        artifact_output_path=artifact_output_path,
    )
    output_record = {
        "flag": "--artifact-output",
        "path": str(output_path),
        "exists": output_path.exists(),
        "artifact_ref": artifact_ref,
    }
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            _download_payload_text(value, output_path),
            encoding="utf-8",
        )
        output_record["exists"] = output_path.exists()

    outputs_written = [str(output_path)] if output_record["exists"] else []
    output_count = 1
    outputs_written_count = len(outputs_written)
    payload = {
        "schema_version": AGENT_LEARNING_CLI_SCHEMA_VERSION,
        "kind": AGENT_LEARNING_ACTION_RUN_KIND,
        "name": str(name or f"{action_id}-action-run"),
        "status": "passed",
        "exit_code": 0,
        "source_path": str(source_path),
        "cwd": str(run_cwd),
        "dry_run": bool(dry_run),
        "action": action,
        "command": f"download {artifact_ref} -> {output_path}",
        "command_args": [],
        "artifact_ref": artifact_ref,
        "artifact_output_path": str(output_path),
        "logs": {"stdout": "", "stderr": "", "stdout_bytes": 0, "stderr_bytes": 0},
        "outputs": [output_record],
        "outputs_written": outputs_written,
        "summary": {
            "action_id": str(action.get("id") or action_id),
            "action_label": action.get("label"),
            "action_kind": "download",
            "source_card_path": action.get("source_card_path"),
            "requires_input": bool(action.get("inputs")),
            "command_exit_code": 0,
            "stdout_bytes": 0,
            "stderr_bytes": 0,
            "output_count": output_count,
            "outputs_written_count": outputs_written_count,
            "output_completion_rate": round(outputs_written_count / output_count, 4),
            "artifact_ref": artifact_ref,
        },
    }
    return payload


def action_catalog(
    artifact: Mapping[str, Any],
    *,
    source_path: str | Path = ".",
    action_id: Optional[str] = None,
    name: Optional[str] = None,
) -> dict[str, Any]:
    artifacts = [artifact, _generic_report_action_artifact(source_path)]
    synthesized = _synthesized_report_artifact(artifact, source_path=source_path)
    if synthesized is not None:
        artifacts.append(synthesized)
    actions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in artifacts:
        for action in extract_actions(item, action_id=action_id):
            key = str(action.get("id") or action.get("command") or action.get("path") or "")
            if key in seen:
                continue
            seen.add(key)
            action["requires_input"] = bool(action.get("inputs"))
            actions.append(action)
    source = normalize_public_payload(artifact)
    source_kind = source.get("kind", source.get("schema_version")) if isinstance(source, Mapping) else None
    source_name = source.get("name") if isinstance(source, Mapping) else None
    cards = sorted({
        str(item.get("source_card_path"))
        for item in actions
        if item.get("source_card_path")
    })
    payload = {
        "schema_version": AGENT_LEARNING_CLI_SCHEMA_VERSION,
        "kind": AGENT_LEARNING_ACTIONS_KIND,
        "name": str(name or source_name or Path(source_path).stem),
        "status": "passed",
        "exit_code": 0,
        "source_path": str(source_path),
        "actions": actions,
        "summary": {
            "source_kind": source_kind,
            "source_name": source_name,
            "action_count": len(actions),
            "action_ids": [str(item.get("id")) for item in actions if item.get("id")],
            "source_card_paths": cards,
        },
    }
    if action_id is not None:
        payload["summary"]["filter_action_id"] = action_id
    return payload


def _generic_report_action_artifact(source_path: str | Path) -> dict[str, Any]:
    return {
        "actions": [
            {
                "id": "report_artifact",
                "kind": "cli",
                "label": "Report artifact",
                "command": "agent-learn report",
                "command_args": [
                    "agent-learn",
                    "report",
                    str(Path(source_path).expanduser().resolve()),
                    "--output",
                    "report-artifact.json",
                ],
                "target_layers": ["reporting", "diagnosis"],
            }
        ]
    }


def _synthesized_report_artifact(
    artifact: Mapping[str, Any],
    *,
    source_path: str | Path,
) -> Optional[dict[str, Any]]:
    try:
        cli = importlib.import_module("fi.alk.simulate.cli")
        report = cli._report_result(
            source=artifact,
            source_path=Path(source_path),
            name=None,
            duration_seconds=0.0,
        )
    except Exception:
        return None
    return report if isinstance(report, dict) else None


def render_markdown(catalog: Mapping[str, Any]) -> str:
    summary = dict(catalog.get("summary") or {})
    rows = [
        "| Action | Label | Source card | Status | Target layers | Command |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for action in catalog.get("actions") or []:
        if not isinstance(action, Mapping):
            continue
        rows.append(
            "| "
            + " | ".join(
                _md_cell(value)
                for value in [
                    action.get("id"),
                    action.get("label"),
                    action.get("source_card_path"),
                    action.get("readiness_status")
                    or action.get("strategy_status")
                    or action.get("diagnosis_status")
                    or action.get("status"),
                    _join_values(action.get("target_layers")),
                    action.get("command") or action.get("artifact_ref"),
                ]
            )
            + " |"
        )
    if len(rows) == 2:
        rows.append("| No actions |  |  |  |  |  |")
    lines = [
        f"# {_md_text(catalog.get('name') or 'artifact-actions')}",
        "",
        f"- Source: `{_md_code(catalog.get('source_path') or '.')}`",
        f"- Source kind: {_md_text(summary.get('source_kind') or 'unknown')}",
        f"- Actions: {_md_text(summary.get('action_count') or 0)}",
        "",
        "## Actions",
        "",
        *rows,
        "",
    ]
    return "\n".join(lines)


def render_action_run_markdown(result: Mapping[str, Any]) -> str:
    rows = [
        "| Output | Exists |",
        "| --- | --- |",
    ]
    for item in result.get("outputs") or []:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            "| "
            + " | ".join(_md_cell(value) for value in [item.get("path"), item.get("exists")])
            + " |"
        )
    if len(rows) == 2:
        rows.append("| No declared outputs |  |")
    summary = dict(result.get("summary") or {})
    logs = dict(result.get("logs") or {})
    lines = [
        f"# {_md_text(result.get('name') or 'action-run')}",
        "",
        f"- Source: `{_md_code(result.get('source_path') or '.')}`",
        f"- Action: {_md_text(summary.get('action_id') or 'unknown')}",
        f"- Action kind: {_md_text(summary.get('action_kind') or 'cli')}",
        f"- Status: {_md_text(result.get('status') or 'unknown')}",
        f"- Exit code: {_md_text(result.get('exit_code'))}",
        f"- Command: `{_md_code(result.get('command') or '')}`",
        "",
        "## Outputs",
        "",
        *rows,
        "",
        "## Logs",
        "",
        f"- Stdout bytes: {_md_text(summary.get('stdout_bytes') or 0)}",
        f"- Stderr bytes: {_md_text(summary.get('stderr_bytes') or 0)}",
        "",
    ]
    stdout = str(logs.get("stdout") or "")
    stderr = str(logs.get("stderr") or "")
    if stdout:
        lines.extend(["### Stdout", "", "```text", stdout.rstrip(), "```", ""])
    if stderr:
        lines.extend(["### Stderr", "", "```text", stderr.rstrip(), "```", ""])
    return "\n".join(lines)


def _walk_actions(value: Any, path: str = "") -> Iterable[tuple[str, Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        if (
            value.get("kind") == "cli"
            and value.get("command_args") is not None
        ) or (
            value.get("kind") == "download"
            and value.get("artifact_ref") is not None
        ):
            yield path, value
            return
        for key, item in value.items():
            item_path = f"{path}.{key}" if path else str(key)
            yield from _walk_actions(item, item_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            item_path = f"{path}.{index}" if path else str(index)
            yield from _walk_actions(item, item_path)


def _source_card_path(action_path: str) -> str:
    marker = ".actions."
    if marker in action_path:
        source = action_path.split(marker, 1)[0]
        return source.removeprefix("report.")
    if action_path.endswith(".actions"):
        return action_path[: -len(".actions")].removeprefix("report.")
    return action_path


def _resolved_command_args(
    action: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> list[str]:
    input_defaults = {
        str(item.get("name")): item.get("default")
        for item in action.get("inputs") or []
        if isinstance(item, Mapping) and item.get("name") not in (None, "")
    }
    values = {**input_defaults, **{str(key): value for key, value in inputs.items()}}
    resolved: list[str] = []
    for raw_arg in action.get("command_args") or []:
        arg = str(raw_arg)
        for name, value in values.items():
            arg = arg.replace("{{" + name + "}}", str(value))
        if "{{" in arg or "}}" in arg:
            raise ValueError(f"action {action.get('id')!r} requires input for {arg}")
        resolved.append(arg)
    return resolved


def _dispatch_action_command(command_args: list[str], *, cwd: Path) -> dict[str, Any]:
    previous_cwd = Path.cwd()
    cwd.mkdir(parents=True, exist_ok=True)
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        os.chdir(cwd)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            cli = importlib.import_module("fi.alk.cli")
            exit_code = int(cli.main(command_args[1:]))
    finally:
        os.chdir(previous_cwd)
    stdout_text = stdout.getvalue()
    stderr_text = stderr.getvalue()
    return {
        "exit_code": exit_code,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "stdout_bytes": len(stdout_text.encode("utf-8")),
        "stderr_bytes": len(stderr_text.encode("utf-8")),
    }


def _command_output_records(command_args: list[str], cwd: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for flag, value in _command_output_values(command_args):
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = cwd / path
        path = path.resolve()
        records.append({"flag": flag, "path": str(path), "exists": path.exists()})
    return records


def _absolutize_output_args(command_args: list[str], cwd: Path) -> list[str]:
    output_flags = {"-o", "--output", "--junit", "--sarif", "--markdown", "--md"}
    resolved = list(command_args)
    index = 0
    while index < len(resolved):
        arg = resolved[index]
        if arg in output_flags and index + 1 < len(resolved):
            resolved[index + 1] = str(_output_arg_path(resolved[index + 1], cwd))
            index += 2
            continue
        replaced = False
        for flag in output_flags:
            prefix = flag + "="
            if arg.startswith(prefix):
                resolved[index] = prefix + str(_output_arg_path(arg[len(prefix):], cwd))
                replaced = True
                break
        index += 1
        if replaced:
            continue
    return resolved


def _command_output_values(command_args: list[str]) -> list[tuple[str, str]]:
    output_flags = {"-o", "--output", "--junit", "--sarif", "--markdown", "--md"}
    values: list[tuple[str, str]] = []
    index = 0
    while index < len(command_args):
        arg = command_args[index]
        flag: Optional[str] = None
        value: Optional[str] = None
        if arg in output_flags and index + 1 < len(command_args):
            flag = arg
            value = command_args[index + 1]
            index += 2
        else:
            for candidate in output_flags:
                prefix = candidate + "="
                if arg.startswith(prefix):
                    flag = candidate
                    value = arg[len(prefix):]
                    break
            index += 1
        if flag is None or value in (None, ""):
            continue
        values.append((flag, str(value)))
    return values


def _output_arg_path(value: str, cwd: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (cwd / path).resolve()


def _resolve_artifact_ref(
    artifact: Mapping[str, Any],
    artifact_ref: str,
    *,
    source_path: str | Path = ".",
) -> Any:
    normalized = normalize_public_payload(artifact)
    if not isinstance(normalized, Mapping):
        return _MISSING
    candidates = [artifact_ref]
    if artifact_ref.startswith("report."):
        candidates.append(artifact_ref.removeprefix("report."))
    artifact_candidates: list[Mapping[str, Any] | None] = [normalized]
    if artifact_ref.startswith("report."):
        artifact_candidates.append(
            _synthesized_report_artifact(
                artifact,
                source_path=source_path,
            )
        )
    for artifact_candidate in artifact_candidates:
        if not isinstance(artifact_candidate, Mapping):
            continue
        resolved = normalize_public_payload(artifact_candidate)
        if not isinstance(resolved, Mapping):
            continue
        for candidate in candidates:
            value = _resolve_path(resolved, candidate.split("."))
            if value is not _MISSING:
                return value
    return _MISSING


def _resolve_path(value: Any, path: list[str]) -> Any:
    current = value
    for part in path:
        if isinstance(current, Mapping):
            if part not in current:
                return _MISSING
            current = current[part]
            continue
        if isinstance(current, list):
            try:
                index = int(part)
            except ValueError:
                return _MISSING
            if index < 0 or index >= len(current):
                return _MISSING
            current = current[index]
            continue
        return _MISSING
    return current


def _download_output_path(
    action: Mapping[str, Any],
    inputs: Mapping[str, Any],
    cwd: Path,
    *,
    artifact_output_path: str | Path | None,
) -> Path:
    raw_value = artifact_output_path
    if raw_value in (None, ""):
        for name in ("artifact_output", "artifact_output_path", "output_path"):
            if inputs.get(name) not in (None, ""):
                raw_value = inputs[name]
                break
    if raw_value in (None, ""):
        raw_value = action.get("default_filename")
    if raw_value in (None, ""):
        raw_value = f"{_slug(action.get('id') or 'artifact')}.json"

    path = Path(str(raw_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (cwd / path).resolve()


def _download_payload_text(value: Any, output_path: Path) -> str:
    if isinstance(value, str) and output_path.suffix.lower() not in {".json", ".jsonl"}:
        return value if value.endswith("\n") else value + "\n"
    return json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"


def _slug(value: Any, default: str = "artifact") -> str:
    text = str(value or default).strip().lower()
    chars = [char if char.isalnum() else "-" for char in text]
    slug = "-".join(part for part in "".join(chars).split("-") if part)
    return slug or default


def _shell_token(value: Any) -> str:
    text = str(value)
    if all(char.isalnum() or char in "-_./:=@" for char in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _load_json_or_yaml(path: Path) -> Any:
    if not path.exists():
        raise ValueError(f"action artifact file not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency clarity
            raise ValueError("YAML artifacts require PyYAML; use JSON or install PyYAML.") from exc
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _join_values(value: Any) -> Optional[str]:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, (list, tuple, set)):
        values = [str(item) for item in value if item not in (None, "")]
        return ", ".join(values) if values else None
    return str(value)


def _md_text(value: Any) -> str:
    return str(value).replace("\n", " ")


def _md_code(value: Any) -> str:
    return str(value).replace("`", "\\`")


def _md_cell(value: Any) -> str:
    text = _md_text(value if value is not None else "").replace("|", "\\|")
    return text if len(text) <= 140 else f"{text[:137]}..."


__all__ = [
    "AGENT_LEARNING_ACTIONS_KIND",
    "AGENT_LEARNING_ACTION_RUN_KIND",
    "action_catalog",
    "extract_actions",
    "get_action",
    "load_artifact_file",
    "render_action_run_markdown",
    "render_markdown",
    "run_action",
]
