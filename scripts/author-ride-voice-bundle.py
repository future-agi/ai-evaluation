#!/usr/bin/env python3
"""Author a futureagi.environment-bundle.v2 for the ride-voice-agent.

Reads agent source (db/schema.sql, db/seed.sql) and a generated session
(scenarios/, contract.json, simulator_prompt.md) and produces a sealed,
preflight-validated V2 bundle directory ready for Daytona upload.

Usage:
  # From repo root (src/ is on sys.path via pyproject):
  python scripts/author-ride-voice-bundle.py \
    --agent-source /path/to/ride-voice-agent \
    --session /path/to/sessions/ride-voice-XYZ \
    --output /tmp/ride-voice-bundle \
    --commit $(git -C /path/to/ride-voice-agent rev-parse HEAD)

  # Validate-only (--dry-run): loads the bundle through load_bundle_v2 + preflight
  # but does not write anything new.
  python scripts/author-ride-voice-bundle.py \
    --output /tmp/ride-voice-bundle --dry-run

Environment:
  Optionally set secret refs for preflight validation:
    HARNESS_CREDS_JSON  Path to a JSON file mapping secret names -> "target_provider".
                        When absent, preflight skips secret validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

# Ensure the repo's src is importable even when run standalone.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fi.alk.harness.bundle import CapabilityProtocol  # noqa: E402
from fi.alk.harness.bundle_v2 import (  # noqa: E402
    BUNDLE_V2_SCHEMA_VERSION,
    BaselineStrategy,
    BundleFileV2,
    BundleProvenanceV2,
    BundleRuntimeV2,
    CapabilityV2,
    EnvironmentBundleV2,
    EvidenceSeam,
    ManagedEngine,
    ManagedProcess,
    ProcessUser,
    ReadinessProbeV2,
    RuntimeKindV2,
    SecretPurpose,
    Seed,
    Sentinel,
    SourceProcess,
    StartedCheck,
    StoreBaseline,
    StoreEntry,
    compute_inputs_digest,
    load_bundle_v2,
    seal_bundle_v2,
)
from fi.alk.harness.process_preflight import preflight_bundle  # noqa: E402


def _file_records(root: Path) -> list[BundleFileV2]:
    """Hash every non-manifest file under the bundle root."""
    records = []
    for p in sorted(root.rglob("*")):
        if p.is_dir() or p.name == "manifest.json":
            continue
        data = p.read_bytes()
        records.append(BundleFileV2(
            path=p.relative_to(root).as_posix(),
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
        ))
    return records


def _copy_scenarios(session_dir: Path, out: Path) -> int:
    """Copy scenario documents into the bundle's guest format."""
    scenarios_src = session_dir / "scenarios"
    if not scenarios_src.is_dir():
        print(f"warning: no scenarios/ in {session_dir}", file=sys.stderr)
        return 0
    (out / "scenarios").mkdir(exist_ok=True)
    count = 0
    for d in sorted(scenarios_src.iterdir()):
        if not d.is_dir():
            continue
        dst = out / "scenarios" / d.name
        (dst / "checks").mkdir(parents=True, exist_ok=True)
        checks = sorted(p.stem for p in (d / "checks").glob("*.py")) if (d / "checks").is_dir() else []
        body = json.loads((d / "scenario.json").read_text(encoding="utf-8"))
        scenario_doc = {
            "scenario_key": body.get("scenario_key") or d.name,
            "scenario_id": "",
            "sub_goals": checks,
            "voice_case": body.get("voice_case") or "2.1.2",
            "instruction": body.get("instruction", ""),
            "tests": body.get("tests", ""),
            "persona": body.get("persona") or {"name": "customer"},
            "fixture": body.get("fixture") or {},
        }
        (dst / "scenario.json").write_text(
            json.dumps(scenario_doc, indent=2) + "\n", encoding="utf-8"
        )
        for src_file in ("setup.py", "ready.py"):
            src_path = d / src_file
            if src_path.is_file():
                shutil.copy(src_path, dst / src_file)
            else:
                (dst / src_file).write_text("", encoding="utf-8")
        if (d / "checks").is_dir():
            for c in sorted((d / "checks").glob("*.py")):
                shutil.copy(c, dst / "checks" / c.name)
        count += 1
    return count


def author(
    agent_source: Path,
    session_dir: Path,
    output: Path,
    commit: str,
    *,
    voice_case: str = "2.1.2",
) -> str:
    """Author the bundle. Returns the sealed digest string."""
    # 1. Fresh bundle dir.
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)

    # 2. Seed + migration files.
    (output / "db").mkdir()
    shutil.copy(agent_source / "db/schema.sql", output / "db/schema.sql")
    shutil.copy(agent_source / "db/seed.sql", output / "db/seed.sql")

    # 3. Scenarios.
    scen_count = _copy_scenarios(session_dir, output)
    print(f"scenarios: {scen_count}")

    # 4. Extra bundle-travelling files.
    for extra in ("contract.json", "simulator_prompt.md"):
        src = session_dir / extra
        if src.is_file():
            shutil.copy(src, output / extra)
            print(f"bundled {extra}")

    # 5. Processes.
    processes = [
        ManagedProcess(
            name="postgres", engine=ManagedEngine.POSTGRES, version="16",
            user=ProcessUser.SVC_DATA, depends_on=[],
        ),
        SourceProcess(
            name="tools-api", working_directory="tools-api",
            build_commands=[
                ["python3.12", "-m", "venv", ".venv"],
                [".venv/bin/pip", "install", "--no-cache-dir", "-r", "requirements.txt"],
            ],
            run_command=[
                ".venv/bin/python", "-c",
                "import os,uvicorn;uvicorn.run('main:app',host='127.0.0.1',port=int(os.environ['PORT']))",
            ],
            environment={"DATABASE_URL": "{{DATABASE_URL}}", "PORT": "{{PORT_tools-api}}"},
            started_check=StartedCheck(port=True, timeout_seconds=90),
            user=ProcessUser.SVC_TOOLS, depends_on=["postgres"],
        ),
        SourceProcess(
            name="agent", working_directory=".",
            build_commands=[
                ["uv", "sync", "--no-dev", "--python", "3.12"],
                ["uv", "run", "--no-sync", "--python", "3.12", "--module", "livekit.agents", "download-files"],
            ],
            run_command=[
                "sh", "-c",
                'printf "%s" "$GOOGLE_APPLICATION_CREDENTIALS_JSON" > "$WORLD_DIR/sa.json"; '
                'export GOOGLE_APPLICATION_CREDENTIALS="$WORLD_DIR/sa.json"; '
                'exec uv run --no-sync --python 3.12 agent/agent.py start',
            ],
            environment={
                "DATABASE_URL": "{{DATABASE_URL}}",
                "TOOLS_API_URL": "{{TOOLS_API_URL}}",
                "WORLD_DIR": "{{WORLD_DIR}}",
                "LIVEKIT_AGENT_NAME": "agent-w{{WORLD_INDEX}}",
                "HARNESS_TOOL_TRACE": "{{WORLD_DIR}}/agent-tool-calls.jsonl",
                "GOOGLE_GENAI_USE_VERTEXAI": "True",
                "AGENT_STT_MODEL": "nova-3",
                "AGENT_STT_MODEL_PHONE": "nova-2-phonecall",
            },
            started_check=StartedCheck(log_marker="registered worker", timeout_seconds=240),
            secret_purposes=[SecretPurpose.TARGET_PROVIDER],
            user=ProcessUser.SVC_AGENT, depends_on=["postgres", "tools-api"],
        ),
    ]

    # 6. Seed store.
    inputs_digest = compute_inputs_digest(
        output, ["db/schema.sql"], ["db/seed.sql"],
        engine=ManagedEngine.POSTGRES, version="16",
    )
    seed = Seed(stores=[
        StoreEntry(
            capability="db",
            migrations=["db/schema.sql"],
            seed_files=["db/seed.sql"],
            baseline=StoreBaseline(
                strategy=BaselineStrategy.TEMPLATE_DATABASE, inputs_digest=inputs_digest,
            ),
            sentinel=Sentinel(query="SELECT count(*)::text FROM users", expected="9"),
        )
    ])

    capabilities = {
        "db": CapabilityV2(
            protocol=CapabilityProtocol.POSTGRES, service="postgres",
            container_port=5432, configuration_name="DATABASE_URL",
        ),
        "tools": CapabilityV2(
            protocol=CapabilityProtocol.HTTP, service="tools-api",
            container_port=8080, configuration_name="TOOLS_API_URL",
        ),
    }
    readiness = [ReadinessProbeV2(capability="tools", path="/health", timeout_seconds=180)]

    # 7. files[] + provenance.
    files = _file_records(output)
    provenance = BundleProvenanceV2(
        source_kind="github",
        repository="future-agi/ride-voice-agent",
        commit=commit,
        source_digest=hashlib.sha256(commit.encode()).hexdigest(),
        generated_files=[f.path for f in files if f.path.startswith("scenarios/")],
        adopted_files=["db/schema.sql", "db/seed.sql"],
    )

    manifest = EnvironmentBundleV2(
        schema_version=BUNDLE_V2_SCHEMA_VERSION,
        digest="sha256:" + "0" * 64,
        name="ride-voice",
        runtime=BundleRuntimeV2(
            kind=RuntimeKindV2.PROCESS, control_service="agent",
            evidence_seam=EvidenceSeam.HTTP_TOOL,
        ),
        processes=processes,
        seed=seed,
        capabilities=capabilities,
        readiness=readiness,
        files=files,
        provenance=provenance,
        metadata={
            "agent_name": "ride-voice",
            "authored_from": f"session {session_dir.name}",
            "voice_case": voice_case,
        },
    )
    digest = seal_bundle_v2(manifest)
    manifest = manifest.model_copy(update={"digest": digest})
    (output / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    print(f"digest: {digest}")
    return digest


def validate(output: Path, *, creds_json_path: str | None = None) -> None:
    """Load + preflight the bundle at output."""
    reloaded = load_bundle_v2(output)
    print("load_bundle_v2: OK")
    secret_refs: dict[str, str] = {}
    if creds_json_path and Path(creds_json_path).is_file():
        with open(creds_json_path, encoding="utf-8") as fh:
            secret_refs = {k: "target_provider" for k in json.load(fh)}
    preflight_bundle(output, reloaded, parallelism=1, secret_refs=secret_refs)
    print("preflight_bundle: OK")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Author a ride-voice V2 bundle from agent source + session.",
    )
    parser.add_argument(
        "--agent-source", type=Path,
        help="Path to the ride-voice-agent checkout (must contain db/schema.sql, db/seed.sql).",
    )
    parser.add_argument(
        "--session", type=Path,
        help="Path to the generated session directory (scenarios/, contract.json, etc.).",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output bundle directory.",
    )
    parser.add_argument(
        "--commit", default="0000000000000000000000000000000000000000",
        help="Git commit SHA for provenance.",
    )
    parser.add_argument(
        "--voice-case", default="2.1.2",
        help="Voice case ID for bundle metadata (default: 2.1.2).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate an existing bundle at --output without authoring.",
    )
    args = parser.parse_args(argv)

    creds_path = os.environ.get("HARNESS_CREDS_JSON", "")

    if args.dry_run:
        if not args.output.is_dir():
            print(f"error: --output {args.output} is not a directory", file=sys.stderr)
            return 1
        validate(args.output, creds_json_path=creds_path)
        print(f"BUNDLE VALID at {args.output}")
        return 0

    if not args.agent_source or not args.session:
        parser.error("--agent-source and --session are required unless --dry-run")

    if not (args.agent_source / "db" / "schema.sql").is_file():
        print(f"error: {args.agent_source}/db/schema.sql not found", file=sys.stderr)
        return 1
    if not (args.session / "scenarios").is_dir():
        print(f"error: {args.session}/scenarios/ not found", file=sys.stderr)
        return 1

    author(
        args.agent_source, args.session, args.output, args.commit,
        voice_case=args.voice_case,
    )
    validate(args.output, creds_json_path=creds_path)
    print(f"BUNDLE READY at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
