from __future__ import annotations

import json
from pathlib import Path

import pytest

from fi.alk.harness.job import (
    HarnessJob,
)
from fi.simulate.runtime.spec import RuntimeRequirements


def _job(**runtime):
    return HarnessJob.model_validate(
        {
            "schema_version": "futureagi.harness-job.v1",
            "job_id": "job-1",
            "run_id": "run-1",
            "execution": "hosted",
            "source": {
                "kind": "github",
                "repository": "future-agi/ride-voice-agent",
                "ref": "main",
                "commit_sha": "a" * 40,
                "visibility": "public",
            },
            "agent": {"connector": "vapi", "config": {}, "secret_refs": {}},
            "scenario_count": 10,
            "seed": 1,
            "runtime": {
                "isolation": "dedicated_vm",
                "cpu_units": 4,
                "memory_mb": 8192,
                "parallelism": 3,
                "concurrency_weight": 1,
                "max_duration_seconds": 3600,
                "network_policy": "live",
                **runtime,
            },
            "security": {
                "untrusted_source": True,
                "read_only_source": True,
                "allow_privileged": False,
                "allow_host_runtime_control": False,
                "allowed_egress_domains": [],
            },
            "artifacts": {
                "level": "full",
                "retention_days": 30,
                "allow_bundle_download": True,
                "max_artifact_bytes": 1024,
            },
        }
    )


def test_runtime_requirements_preserve_hosted_parallelism():
    runtime = RuntimeRequirements.model_validate({"parallelism": 8})
    assert runtime.parallelism == 8


def test_hosted_job_rejects_parallelism_above_cpu():
    with pytest.raises(ValueError, match="hosted_parallelism_exceeds_cpu"):
        _job(cpu_units=2, parallelism=3)


def test_hosted_job_requires_resolved_commit():
    payload = _job().model_dump(mode="json")
    payload["source"]["commit_sha"] = None
    with pytest.raises(ValueError, match="github_commit_sha_required"):
        HarnessJob.model_validate(payload)


def test_snapshot_catalog_matches_engine_contract():
    root = Path(__file__).parents[1]
    catalog = json.loads(
        (root / "hosted-snapshot" / "catalog.json").read_text(encoding="utf-8")
    )
    assert catalog["runtimes"] == {
        "python": ["3.11", "3.12", "3.13"],
        "node": ["20", "22"],
    }
    assert catalog["binaries"] == ["git", "ffmpeg"]
    assert {engine: data["version"] for engine, data in catalog["engines"].items()} == {
        "postgres": "16",
        "redis": "7",
        "rabbitmq": "3.13",
    }
