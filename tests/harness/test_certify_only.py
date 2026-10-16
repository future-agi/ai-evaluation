from __future__ import annotations

import asyncio
from functools import partial
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from fi.alk.harness import authoring_runtime_validation as runtime_validation
from fi.alk.harness import bundle_author_v2 as bundle_author
from fi.alk.harness import hosted_authoring_entrypoint as entrypoint
from fi.alk.harness import authoring_entrypoint
from fi.alk.harness.authoring_entrypoint import (
    _prepare_source_data_invariants,
    main as authoring_main,
)
from fi.alk.harness.authoring_runtime_validation import (
    RuntimeValidationError,
    _artifact_digest,
)
from fi.alk.harness.bundle_v2 import load_bundle_v2
from fi.alk.harness.catalogue import Catalogue, SubGoal
from fi.alk.harness.certification import (
    CertificationChecks,
    CheckStatus,
    GenericHarnessArtifactStore,
    RuntimeValidationEvidence,
)
from fi.alk.harness.diagnostics import HarnessDiagnostic
from fi.alk.harness.job import HarnessJob, HarnessStage
from fi.alk.harness.provision import source_fingerprint
from fi.alk.harness.scenario_source import load_scenarios
from fi.alk.harness.source_data_invariants import (
    ARTIFACT,
    author_invariants,
    prune_invariant_scopes,
)
from fi.alk.harness.source_model import SourceModel
from fi.alk.harness.world_ir import WorldIR

_CHECK = "def check(world, calls):\n    return None\n"


def _job(scenario_count: int) -> HarnessJob:
    return HarnessJob.model_validate(
        {
            "job_id": "job-certify",
            "run_id": "run-certify",
            "execution": "hosted",
            "source": {"kind": "archive", "archive_artifact_id": "source-1"},
            "agent": {"connector": "http"},
            "scenario_count": scenario_count,
            "metadata": {"generic_harness_v1": True},
            "runtime": {
                "isolation": "dedicated_vm",
                "cpu_units": 2,
                "memory_mb": 4096,
                "parallelism": 1,
            },
        }
    )


def _scenario(
    authoring: Path, name: str, sub_goals: list[str] | None, **document: object
) -> None:
    """``sub_goals=None`` writes a document without the field, as older suites have."""
    folder = authoring / "scenarios" / str(document.pop("folder", name))
    (folder / "checks").mkdir(parents=True)
    if sub_goals is not None:
        document["sub_goals"] = sub_goals
    (folder / "scenario.json").write_text(
        json.dumps({"name": name, "instruction": f"Test {name}", **document}),
        encoding="utf-8",
    )
    (folder / "setup.py").write_text("def setup(world):\n    return None\n")
    (folder / "ready.py").write_text("def ready(world):\n    return None\n")
    for goal in sub_goals or []:
        (folder / "checks" / f"{goal}.py").write_text("# from another environment\n")


def _assembled(
    tmp_path: Path,
    scenarios: dict[str, list[str]],
    *,
    catalogue: Catalogue | None = None,
) -> tuple[Path, Path, Path]:
    """An environment's files plus a suite's scenarios, as the platform uploads them."""
    source = tmp_path / "source"
    (source / "db").mkdir(parents=True)
    (source / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    (source / "db" / "schema.sql").write_text(
        "CREATE TABLE users (id TEXT PRIMARY KEY);\n", encoding="utf-8"
    )
    authoring = tmp_path / "authoring"
    authoring.mkdir()
    (authoring / "contract.json").write_text(
        json.dumps({"modality": "chat", "tools": []}), encoding="utf-8"
    )
    (authoring / "sub_goals.json").write_text(
        (
            catalogue or Catalogue(sub_goals=[SubGoal(name="works", check=_CHECK)])
        ).model_dump_json(),
        encoding="utf-8",
    )
    artifact_root = authoring / "generic-harness"
    artifact_root.mkdir()
    source_model = SourceModel.create(
        source_digest="sha256:" + source_fingerprint(source), engine="postgres"
    )
    world = WorldIR.create(source_model_fingerprint=source_model.fingerprint, tables=())
    (artifact_root / "source-model.json").write_text(
        source_model.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (artifact_root / "world-ir.json").write_text(
        world.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    for name, goals in scenarios.items():
        _scenario(authoring, name, goals)
    job_path = tmp_path / "certify-job.json"
    job_path.write_text(
        _job(max(1, len(scenarios))).model_dump_json(indent=2), encoding="utf-8"
    )
    return source, authoring, job_path


def _certifying_validate(calls: list[str]):
    """Stand-in for the runtime gate: build, record evidence and freeze, as validate_once does."""

    async def validate(job, source, authoring):
        calls.append("validate")
        bundle = authoring.parent / "validation-bundle"
        manifest = bundle_author.author_bundle_v2(
            source=source, job=job, authoring=authoring, output=bundle
        )
        store = GenericHarnessArtifactStore(authoring / "generic-harness")
        store.write_runtime_evidence(
            RuntimeValidationEvidence(
                source_digest=source_fingerprint(source),
                source_schema_hash=store.read_source_model().fingerprint,
                world_ir_hash=store.read_world_ir().fingerprint,
                compiler_version="test-compiler",
                bundle_digest=manifest.digest,
                contract_hash=_artifact_digest(authoring / "contract.json"),
                scenario_set_hash=_artifact_digest(authoring / "scenarios"),
                checks=CertificationChecks(
                    static=CheckStatus.PASSED,
                    schema_and_seed=CheckStatus.PASSED,
                    processes=CheckStatus.PASSED,
                    source_invariants=CheckStatus.PASSED,
                    scenario_setup_ready=f"{job.scenario_count}/{job.scenario_count}",
                    tool_contract="0/0",
                    action_behavior="0/0",
                    reset_equivalence=CheckStatus.PASSED,
                    world_isolation=CheckStatus.PASSED,
                ),
            )
        )
        runtime_validation._freeze_certified_bundle(bundle, authoring)
        shutil.rmtree(bundle)
        return job.scenario_count

    return validate


def _use_validator(monkeypatch, validate) -> list[dict]:
    seen: list[dict] = []
    real = runtime_validation.validate_and_repair

    async def with_validator(*args, **kwargs):
        seen.append(kwargs)
        return await real(*args, **{**kwargs, "validate": validate})

    monkeypatch.setattr(runtime_validation, "validate_and_repair", with_validator)
    return seen


def _forbid_models(monkeypatch) -> list[str]:
    """Record any model entry point reached; a raise alone would be swallowed into exit 78."""
    from fi.alk.harness import (
        cli,
        repair_authoring,
        runtime_repair,
        session,
        source_data_invariants,
    )

    opened: list[str] = []
    for module, name in (
        (session, "resolve"),
        (cli, "_auto"),
        (cli, "_build"),
        (cli, "_scenarios"),
        (repair_authoring, "request_world_ir_patch"),
        (runtime_repair, "request_runtime_plan_patch"),
        (source_data_invariants, "Stage"),
        (source_data_invariants, "tool_server"),
        (authoring_entrypoint, "inspect_provider_target"),
    ):

        def forbidden(*args, _name=name, **kwargs):
            opened.append(_name)
            raise AssertionError("certify-only must not start a model session")

        monkeypatch.setattr(module, name, forbidden)
    return opened


def _hosted(tmp_path: Path, monkeypatch, argv: list[str]) -> int:
    secrets = tmp_path / "secrets.json"
    secrets.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(entrypoint, "_SECRETS_PATH", secrets)
    monkeypatch.setattr(entrypoint, "_SIMULATOR_SECRETS_PATH", tmp_path / "none.json")
    monkeypatch.setattr(entrypoint, "_ADC_PATH", tmp_path / "adc.json")
    monkeypatch.setattr(entrypoint, "_TARGET_SECRETS_PATH", tmp_path / "target.json")
    return entrypoint.main(argv)


def _argv(source: Path, authoring: Path, job_path: Path, bundle: Path) -> list[str]:
    return [
        str(job_path),
        "--source",
        str(source),
        "--output",
        str(authoring),
        "--certify-only",
        "--bundle-output",
        str(bundle),
    ]


def test_certify_only_certifies_and_seals_the_exact_validated_bundle(
    tmp_path, monkeypatch
) -> None:
    source, authoring, job_path = _assembled(tmp_path, {"one": ["works"]})
    opened = _forbid_models(monkeypatch)
    calls: list[str] = []
    seen = _use_validator(monkeypatch, _certifying_validate(calls))
    bundle = tmp_path / "bundle"

    assert (
        _hosted(tmp_path, monkeypatch, _argv(source, authoring, job_path, bundle)) == 0
    )

    assert calls == ["validate"]
    budgets = seen[0]["controller"].budgets
    assert (
        budgets.compiler_candidates,
        budgets.environment_patches,
        budgets.scenario_patches,
    ) == (0, 0, 0)
    history = json.loads(
        (authoring / "generic-harness" / "repair-history.json").read_text()
    )
    assert [item["action"] for item in history["decisions"]] == ["certify"]
    certificate = json.loads((authoring / "runtime-validation.json").read_text())
    assert certificate["status"] == "certified"
    frozen = load_bundle_v2(authoring / "generic-harness" / "certified-bundle")
    assert load_bundle_v2(bundle).digest == frozen.digest
    assert (
        (authoring / "scenarios" / "one" / "checks" / "works.py")
        .read_text()
        .startswith("def check(world, calls):")
    )
    assert (bundle / "manifest.json").is_file()
    assert (bundle.parent / "runtime-validation.json").is_file()
    assert json.loads((authoring / "certify-report.json").read_text()) == {
        "dropped": [],
        "narrowed": {},
    }
    assert opened == []


def test_certify_only_rejection_exits_78_without_repair(
    tmp_path, monkeypatch, capsys
) -> None:
    source, authoring, job_path = _assembled(tmp_path, {"one": ["works"]})
    opened = _forbid_models(monkeypatch)
    calls: list[str] = []
    diagnostic = HarnessDiagnostic.create(
        stage=HarnessStage.VALIDATING_ENVIRONMENT,
        component="world_ir",
        code="required_value_missing",
        message="missing value",
    )

    async def rejecting(*args):
        calls.append("validate")
        raise RuntimeValidationError(
            "environment", "invalid", diagnostics=(diagnostic,)
        )

    _use_validator(monkeypatch, rejecting)
    bundle = tmp_path / "bundle"

    assert (
        _hosted(tmp_path, monkeypatch, _argv(source, authoring, job_path, bundle)) == 78
    )

    assert calls == ["validate"]
    history = json.loads(
        (authoring / "generic-harness" / "repair-history.json").read_text()
    )
    assert [item["action"] for item in history["decisions"]] == ["reject"]
    assert history["results"] == []
    assert not bundle.exists()
    assert not (authoring / "certify-report.json").exists()
    assert "RuntimeValidationError: invalid" in capsys.readouterr().err
    assert opened == []


def test_certify_only_grades_only_catalogue_sub_goals_and_persists_count(
    tmp_path, monkeypatch, capsys
) -> None:
    source, authoring, job_path = _assembled(
        tmp_path, {"one": ["works", "gone"], "two": ["gone"]}
    )
    opened = _forbid_models(monkeypatch)
    calls: list[str] = []
    _use_validator(monkeypatch, _certifying_validate(calls))
    bundle = tmp_path / "bundle"

    assert (
        _hosted(tmp_path, monkeypatch, _argv(source, authoring, job_path, bundle)) == 0
    )

    assert sorted(path.name for path in (authoring / "scenarios").iterdir()) == ["one"]
    one = json.loads((authoring / "scenarios" / "one" / "scenario.json").read_text())
    assert one["sub_goals"] == ["works"]
    assert sorted(
        path.name for path in (authoring / "scenarios" / "one" / "checks").iterdir()
    ) == ["works.py"]
    index = json.loads((authoring / "scenarios.json").read_text())
    assert [item["name"] for item in index] == ["one"]
    assert json.loads(job_path.read_text())["scenario_count"] == 1
    assert sum(1 for path in (bundle / "scenarios").iterdir() if path.is_dir()) == 1
    err = capsys.readouterr().err
    assert "one: sub_goals not in the environment catalogue: gone" in err
    assert "two: dropped, no sub_goal left to grade" in err
    assert json.loads((authoring / "certify-report.json").read_text()) == {
        "dropped": ["two"],
        "narrowed": {"one": ["gone"]},
    }
    assert opened == []


def test_certify_only_leaves_scenarios_without_a_sub_goals_field_untouched(
    tmp_path, monkeypatch
) -> None:
    source, authoring, job_path = _assembled(tmp_path, {"one": ["works"]})
    _scenario(authoring, "legacy", None)
    legacy = authoring / "scenarios" / "legacy"
    (legacy / "checks" / "kept.py").write_text("# authored with the suite\n")
    before = (legacy / "scenario.json").read_text()
    opened = _forbid_models(monkeypatch)
    _use_validator(monkeypatch, _certifying_validate([]))
    bundle = tmp_path / "bundle"

    assert (
        _hosted(tmp_path, monkeypatch, _argv(source, authoring, job_path, bundle)) == 0
    )

    assert (legacy / "scenario.json").read_text() == before
    assert sorted(path.name for path in (legacy / "checks").iterdir()) == ["kept.py"]
    index = json.loads((authoring / "scenarios.json").read_text())
    assert [item["name"] for item in index] == ["legacy", "one"]
    assert sum(1 for path in (bundle / "scenarios").iterdir() if path.is_dir()) == 2
    assert json.loads((authoring / "certify-report.json").read_text()) == {
        "dropped": [],
        "narrowed": {},
    }
    assert opened == []


def test_certify_only_drops_a_scenario_with_an_empty_sub_goals_list(
    tmp_path, monkeypatch
) -> None:
    source, authoring, job_path = _assembled(tmp_path, {"one": ["works"], "empty": []})
    opened = _forbid_models(monkeypatch)
    _use_validator(monkeypatch, _certifying_validate([]))
    bundle = tmp_path / "bundle"

    assert (
        _hosted(tmp_path, monkeypatch, _argv(source, authoring, job_path, bundle)) == 0
    )

    assert sorted(path.name for path in (authoring / "scenarios").iterdir()) == ["one"]
    assert json.loads((authoring / "certify-report.json").read_text()) == {
        "dropped": ["empty"],
        "narrowed": {},
    }
    assert opened == []


def test_certify_only_rejects_a_suite_with_nothing_left_to_grade(
    tmp_path, monkeypatch
) -> None:
    source, authoring, job_path = _assembled(tmp_path, {"one": ["gone"]})
    opened = _forbid_models(monkeypatch)
    calls: list[str] = []
    _use_validator(monkeypatch, _certifying_validate(calls))

    assert (
        _hosted(
            tmp_path,
            monkeypatch,
            _argv(source, authoring, job_path, tmp_path / "bundle"),
        )
        == 78
    )
    assert calls == []
    assert opened == []


def test_data_free_review_is_regenerated_after_the_contract_changes(tmp_path) -> None:
    source, authoring, _ = _assembled(tmp_path, {"one": ["works"]})
    artifact = authoring / ARTIFACT
    artifact.write_text(
        json.dumps(
            {
                "status": "not_applicable",
                "reason": "No custom tools, data-store seam, dependencies or runtime business tables",
                "checks": [],
                "tool_execution_proven": False,
                "contract_sha256": "0" * 64,
                "source_sha256": {},
            }
        ),
        encoding="utf-8",
    )

    class EmptyWorld:
        def state(self):
            return {}

    with pytest.raises(ValueError, match="changed after data-free review"):
        asyncio.run(author_invariants(source, authoring, EmptyWorld()))

    assert _prepare_source_data_invariants(_job(1), authoring) == []
    assert not artifact.exists()
    assert asyncio.run(author_invariants(source, authoring, EmptyWorld())) == []
    assert json.loads(artifact.read_text())["contract_sha256"] != "0" * 64


def test_data_ful_review_scopes_are_pruned_to_scenarios_present(tmp_path) -> None:
    _, authoring, _ = _assembled(tmp_path, {"one": ["works"]})
    evidence = [{"path": "agent.py", "quote": "x" * 20, "sha256": "0" * 64}]
    (authoring / ARTIFACT).write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "name": "kept",
                        "violations_sql": "SELECT 1 WHERE false",
                        "evidence": evidence,
                        "scenarios": ["one", "two"],
                    },
                    {
                        "name": "orphaned",
                        "violations_sql": "SELECT 1 WHERE false",
                        "evidence": evidence,
                        "scenarios": ["two"],
                    },
                    {
                        "name": "global",
                        "violations_sql": "SELECT 1 WHERE false",
                        "evidence": evidence,
                        "scenarios": [],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    assert prune_invariant_scopes(authoring) == ["orphaned"]

    checks = json.loads((authoring / ARTIFACT).read_text())["checks"]
    assert [(check["name"], check["scenarios"]) for check in checks] == [
        ("kept", ["one"]),
        ("global", []),
    ]


def test_certify_only_bundle_failure_exits_78(tmp_path, monkeypatch, capsys) -> None:
    source, authoring, job_path = _assembled(tmp_path, {"one": ["works"]})
    opened = _forbid_models(monkeypatch)
    _use_validator(monkeypatch, _certifying_validate([]))

    bundle = tmp_path / "bundle"

    def broken(**kwargs):
        if kwargs["output"] != bundle:
            return None
        raise bundle_author.BundleAuthorError(
            "certified_bundle_inputs_changed: scenarios"
        )

    monkeypatch.setattr(bundle_author, "_reuse_certified_bundle", broken)

    assert (
        _hosted(
            tmp_path,
            monkeypatch,
            _argv(source, authoring, job_path, tmp_path / "bundle"),
        )
        == 78
    )
    assert (
        "Certified bundle could not be sealed: BundleAuthorError: "
        "certified_bundle_inputs_changed: scenarios"
    ) in capsys.readouterr().err
    assert not (authoring / "certify-report.json").exists()
    assert opened == []


def test_certify_only_writes_the_report_before_sealing(tmp_path, monkeypatch) -> None:
    source, authoring, job_path = _assembled(tmp_path, {"one": ["works"]})
    opened = _forbid_models(monkeypatch)
    _use_validator(monkeypatch, _certifying_validate([]))
    bundle = tmp_path / "bundle"
    real = bundle_author._reuse_certified_bundle
    at_seal: list[tuple[bool, bool]] = []

    def sealing(**kwargs):
        if kwargs["output"] == bundle:
            at_seal.append(
                (
                    (authoring / "certify-report.json").is_file(),
                    (bundle.parent / "runtime-validation.json").exists(),
                )
            )
        return real(**kwargs)

    monkeypatch.setattr(bundle_author, "_reuse_certified_bundle", sealing)

    assert (
        _hosted(tmp_path, monkeypatch, _argv(source, authoring, job_path, bundle)) == 0
    )

    assert at_seal == [(True, False)]
    assert (bundle.parent / "runtime-validation.json").is_file()
    assert opened == []


def test_certify_only_unremovable_stale_report_exits_78(
    tmp_path, monkeypatch, capsys
) -> None:
    source, authoring, job_path = _assembled(tmp_path, {"one": ["works"]})
    (authoring / "certify-report.json").mkdir()
    opened = _forbid_models(monkeypatch)
    calls: list[str] = []
    _use_validator(monkeypatch, _certifying_validate(calls))

    assert (
        _hosted(
            tmp_path,
            monkeypatch,
            _argv(source, authoring, job_path, tmp_path / "bundle"),
        )
        == 78
    )
    assert "Certify-only could not clear stale files" in capsys.readouterr().err
    assert calls == []
    assert opened == []


def test_certify_only_without_a_frozen_bundle_exits_78(
    tmp_path, monkeypatch, capsys
) -> None:
    source, authoring, job_path = _assembled(tmp_path, {"one": ["works"]})
    opened = _forbid_models(monkeypatch)

    async def passing_without_freeze(job, *args):
        return job.scenario_count

    _use_validator(monkeypatch, passing_without_freeze)
    bundle = tmp_path / "bundle"

    assert (
        _hosted(tmp_path, monkeypatch, _argv(source, authoring, job_path, bundle)) == 78
    )
    assert "validation left no certified bundle to seal" in capsys.readouterr().err
    assert not bundle.exists()
    assert not (authoring / "certify-report.json").exists()
    assert opened == []


def test_certify_only_requires_bundle_output(tmp_path) -> None:
    with pytest.raises(SystemExit) as exited:
        authoring_main(
            [
                str(tmp_path / "job.json"),
                "--source",
                str(tmp_path),
                "--output",
                str(tmp_path),
                "--certify-only",
            ]
        )
    assert exited.value.code == 2


def test_certify_only_removes_checks_for_sub_goals_this_environment_judges(
    tmp_path, monkeypatch
) -> None:
    catalogue = Catalogue(
        sub_goals=[
            SubGoal(name="works", check=_CHECK),
            SubGoal(name="opinion", judged="Pass: polite. Fail: rude."),
        ]
    )
    source, authoring, job_path = _assembled(
        tmp_path, {"one": ["works", "opinion"]}, catalogue=catalogue
    )
    opened = _forbid_models(monkeypatch)
    _use_validator(monkeypatch, _certifying_validate([]))
    bundle = tmp_path / "bundle"

    assert (
        _hosted(tmp_path, monkeypatch, _argv(source, authoring, job_path, bundle)) == 0
    )

    checks = authoring / "scenarios" / "one" / "checks"
    assert sorted(path.name for path in checks.iterdir()) == ["works.py"]
    [scenario] = load_scenarios(bundle)
    assert {goal.name: bool(goal.judged) for goal in scenario.sub_goals} == {
        "works": False,
        "opinion": True,
    }
    assert opened == []


def test_certify_only_keeps_folder_names_and_scenario_keys(
    tmp_path, monkeypatch
) -> None:
    source, authoring, job_path = _assembled(tmp_path, {})
    _scenario(authoring, "book_ride", ["works"], folder="Old_Folder", extra_field=1)
    _scenario(authoring, "cancel ride", ["works"], folder="f2", scenario_key="key-2")
    opened = _forbid_models(monkeypatch)
    _use_validator(monkeypatch, _certifying_validate([]))
    bundle = tmp_path / "bundle"

    assert (
        _hosted(tmp_path, monkeypatch, _argv(source, authoring, job_path, bundle)) == 0
    )

    assert sorted(path.name for path in (authoring / "scenarios").iterdir()) == [
        "Old_Folder",
        "f2",
    ]
    document = json.loads(
        (authoring / "scenarios" / "Old_Folder" / "scenario.json").read_text()
    )
    assert (document["scenario_key"], document["extra_field"]) == ("book_ride", 1)
    assert sorted(scenario.scenario_key for scenario in load_scenarios(bundle)) == [
        "book_ride",
        "key-2",
    ]
    assert opened == []


@pytest.mark.parametrize(
    "documents",
    [
        [("a", {"scenario_key": "same"}), ("b", {"scenario_key": "same"})],
        [("same", {"folder": "f1"}), ("same", {"folder": "f2"})],
    ],
)
def test_certify_only_rejects_scenarios_sharing_a_key(
    tmp_path, monkeypatch, capsys, documents
) -> None:
    source, authoring, job_path = _assembled(tmp_path, {})
    for name, document in documents:
        _scenario(authoring, name, ["works"], **document)
    opened = _forbid_models(monkeypatch)
    calls: list[str] = []
    _use_validator(monkeypatch, _certifying_validate(calls))

    assert (
        _hosted(
            tmp_path,
            monkeypatch,
            _argv(source, authoring, job_path, tmp_path / "bundle"),
        )
        == 78
    )
    assert "share scenario_key 'same'" in capsys.readouterr().err
    assert calls == []
    assert opened == []


@pytest.mark.parametrize("name", ["../escape", "..", "a/b"])
def test_certify_only_rejects_a_scenario_name_outside_scenarios(
    tmp_path, monkeypatch, capsys, name
) -> None:
    source, authoring, job_path = _assembled(tmp_path, {"one": ["works"]})
    _scenario(authoring, name, ["works"], folder="escaping")
    opened = _forbid_models(monkeypatch)
    _use_validator(monkeypatch, _certifying_validate([]))

    assert (
        _hosted(
            tmp_path,
            monkeypatch,
            _argv(source, authoring, job_path, tmp_path / "bundle"),
        )
        == 78
    )
    assert f"escaping: invalid scenario name {name!r}" in capsys.readouterr().err
    assert not (tmp_path / "escape").exists()
    assert opened == []


@pytest.mark.parametrize(
    ("catalogue", "message"),
    [
        (None, "sub_goals.json is missing from the environment"),
        ("", "sub_goals.json is empty"),
        ("{}", "sub_goals.json defines no sub_goals"),
    ],
)
def test_certify_only_requires_a_sub_goal_catalogue(
    tmp_path, monkeypatch, capsys, catalogue, message
) -> None:
    source, authoring, job_path = _assembled(tmp_path, {"one": ["works"]})
    if catalogue is None:
        (authoring / "sub_goals.json").unlink()
    else:
        (authoring / "sub_goals.json").write_text(catalogue, encoding="utf-8")
    _forbid_models(monkeypatch)

    assert (
        _hosted(
            tmp_path,
            monkeypatch,
            _argv(source, authoring, job_path, tmp_path / "bundle"),
        )
        == 78
    )
    assert message in capsys.readouterr().err


def test_certify_only_invalid_job_exits_78(tmp_path, monkeypatch, capsys) -> None:
    source, authoring, job_path = _assembled(tmp_path, {"one": ["works"]})
    document = json.loads(job_path.read_text())
    job_path.write_text(json.dumps({**document, "scenario_count": 0}))
    _forbid_models(monkeypatch)

    assert (
        _hosted(
            tmp_path,
            monkeypatch,
            _argv(source, authoring, job_path, tmp_path / "bundle"),
        )
        == 78
    )
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1
    assert "Certify-only: invalid job document: ValidationError:" in err[0]


def test_certify_only_fails_when_no_saved_invariant_still_applies(
    tmp_path, monkeypatch, capsys
) -> None:
    source, authoring, job_path = _assembled(
        tmp_path, {"one": ["works"], "two": ["gone"]}
    )
    evidence = [{"path": "agent.py", "quote": "x" * 20, "sha256": "0" * 64}]
    (authoring / ARTIFACT).write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "name": "orphaned",
                        "violations_sql": "SELECT 1 WHERE false",
                        "evidence": evidence,
                        "scenarios": ["two"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    opened = _forbid_models(monkeypatch)
    calls: list[str] = []
    _use_validator(monkeypatch, _certifying_validate(calls))

    assert (
        _hosted(
            tmp_path,
            monkeypatch,
            _argv(source, authoring, job_path, tmp_path / "bundle"),
        )
        == 78
    )
    assert (
        "no saved source-data invariant applies to the scenarios left in the suite"
        in capsys.readouterr().err
    )
    assert calls == []
    assert opened == []


def test_certify_only_never_reads_provider_credentials(tmp_path, monkeypatch) -> None:
    source, authoring, job_path = _assembled(tmp_path, {"one": ["works"]})
    opened = _forbid_models(monkeypatch)
    _use_validator(monkeypatch, _certifying_validate([]))
    target = tmp_path / "target-secrets.json"
    target.write_text('{"VAPI_API_KEY": "secret"}', encoding="utf-8")
    argv = _argv(source, authoring, job_path, tmp_path / "bundle") + [
        "--target-secrets",
        str(target),
        "--provider-profile-cache",
        str(tmp_path / "absent-profile.json"),
    ]

    assert _hosted(tmp_path, monkeypatch, argv) == 0

    assert not target.exists()
    assert not (tmp_path / "absent-profile.json").exists()
    assert opened == []


def _real_runtime(tmp_path: Path, monkeypatch, state: dict) -> None:
    """Fake only the process boundary; validate_once and the invariant review stay real."""
    from fi.alk.harness import hosted_entrypoint, outbound, process_runtime

    secrets = tmp_path / "runtime-secrets.json"
    secrets.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        runtime_validation,
        "validate_once",
        partial(runtime_validation.validate_once, secrets_path=secrets),
    )
    monkeypatch.setattr(
        outbound,
        "load_capabilities",
        lambda *, unlink: SimpleNamespace(attempt_id="test", expires_at=None),
    )

    class Provider:
        def __init__(self, **kwargs):
            pass

        async def provision(self, *args, work_directory, **kwargs):
            artifacts = work_directory / "artifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            (artifacts / "build.json").write_text('{"conformance": true}')
            return [SimpleNamespace(endpoints={})]

        async def reset(self, *args, **kwargs):
            return None

        async def close(self, **kwargs):
            return None

    class World:
        def read_only(self):
            return self

        def state(self):
            return state

        def query(self, sql):
            return []

        def checkpoint(self):
            return "baseline"

        def revert(self, checkpoint):
            return None

    class Factory:
        def __init__(self, work):
            pass

        async def create(self, *args, **kwargs):
            return World()

    monkeypatch.setattr(process_runtime, "ProcessRuntimeProvider", Provider)
    monkeypatch.setattr(hosted_entrypoint, "ProcessWorldFactory", Factory)


def test_certify_only_runs_the_real_runtime_gate_without_a_model(
    tmp_path, monkeypatch
) -> None:
    source, authoring, job_path = _assembled(tmp_path, {"one": ["works"]})
    opened = _forbid_models(monkeypatch)
    _real_runtime(tmp_path, monkeypatch, state={})
    bundle = tmp_path / "bundle"

    assert (
        _hosted(tmp_path, monkeypatch, _argv(source, authoring, job_path, bundle)) == 0
    )

    assert opened == []
    certificate = json.loads((authoring / "runtime-validation.json").read_text())
    assert certificate["status"] == "certified"
    frozen = load_bundle_v2(authoring / "generic-harness" / "certified-bundle")
    assert load_bundle_v2(bundle).digest == frozen.digest
    assert (bundle.parent / "runtime-validation.json").is_file()
    assert json.loads((authoring / ARTIFACT).read_text())["status"] == "not_applicable"


def test_certify_only_refuses_to_author_invariants_for_a_live_world_with_data(
    tmp_path, monkeypatch, capsys
) -> None:
    source, authoring, job_path = _assembled(tmp_path, {"one": ["works"]})
    opened = _forbid_models(monkeypatch)
    _real_runtime(tmp_path, monkeypatch, state={"users": [{"id": "u1"}]})
    bundle = tmp_path / "bundle"

    assert (
        _hosted(tmp_path, monkeypatch, _argv(source, authoring, job_path, bundle)) == 78
    )

    assert opened == []
    assert "the live world has business tables" in capsys.readouterr().err
    history = json.loads(
        (authoring / "generic-harness" / "repair-history.json").read_text()
    )
    assert [item["action"] for item in history["decisions"]] == ["reject"]
    assert not bundle.exists()
    assert not (authoring / ARTIFACT).exists()
