"""Shape of the release workflow: what runs when, in what order, with which secrets."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = PROJECT_ROOT / ".github" / "workflows"
# PyPI's trusted publisher accepts uploads from this file name only.
RELEASE_WORKFLOW = WORKFLOWS / "publish-pypi.yml"
SDK_JOBS = {"sdk-check", "sdk-build", "sdk-publish"}

TEXT = RELEASE_WORKFLOW.read_text(encoding="utf-8")
WORKFLOW = yaml.safe_load(TEXT)
JOBS = WORKFLOW["jobs"]
# PyYAML reads the bare key `on` as the boolean True.
TRIGGERS = WORKFLOW.get("on", WORKFLOW.get(True))


def _needs(job: str) -> set[str]:
    needs = JOBS[job].get("needs", [])
    return {needs} if isinstance(needs, str) else set(needs)


def _dump(job: str) -> str:
    return yaml.safe_dump(JOBS[job])


def _steps(job: str) -> list[dict]:
    return JOBS[job]["steps"]


def test_one_release_workflow():
    assert not (WORKFLOWS / "e2b-template.yml").exists()
    assert WORKFLOW["name"] == "Release"
    assert set(JOBS) == SDK_JOBS | {
        "validate",
        "release-pr",
        "template-publish",
        "deployment-bump",
    }


def test_triggers():
    assert TRIGGERS["push"]["branches"] == ["main"]
    assert "workflow_dispatch" in TRIGGERS
    paths = TRIGGERS["pull_request"]["paths"]
    for path in (
        ".github/workflows/publish-pypi.yml",
        "scripts/e2b-template.py",
        "scripts/bump_deployment_values.py",
    ):
        assert path in paths


def test_sdk_jobs_run_on_push_only():
    assert JOBS["sdk-check"]["if"] == "github.event_name == 'push'"
    assert _needs("sdk-build") == {"sdk-check"}
    assert _needs("sdk-publish") == {"sdk-check", "sdk-build"}
    assert JOBS["sdk-publish"]["environment"] == "pypi"
    assert JOBS["sdk-publish"]["permissions"] == {
        "id-token": "write",
        "contents": "read",
    }


def test_uploads_and_template_builds_wait_for_approval():
    gated = {
        name: job["environment"] for name, job in JOBS.items() if "environment" in job
    }
    # Separate groups: PyPI trusts uploads from any job that runs in `pypi`.
    assert gated == {"sdk-publish": "pypi", "template-publish": "e2b-template"}


def test_template_does_not_wait_for_the_sdk_upload():
    assert _needs("template-publish") == {"validate", "sdk-check"}
    assert _needs("deployment-bump").isdisjoint(SDK_JOBS)
    for job in SDK_JOBS:
        assert _needs(job).isdisjoint(
            {"validate", "template-publish", "deployment-bump", "release-pr"}
        )


def test_template_builds_only_for_a_release_or_a_manual_run():
    assert JOBS["template-publish"]["if"] == (
        "${{ !cancelled() && needs.validate.result == 'success'"
        " && github.ref == 'refs/heads/main'"
        " && (github.event_name == 'workflow_dispatch'"
        " || needs.sdk-check.outputs.publish == 'true') }}"
    )


def test_bump_pull_request_only_when_the_version_is_published():
    assert _needs("release-pr") == {"sdk-check"}
    assert JOBS["release-pr"]["if"] == "needs.sdk-check.outputs.publish == 'false'"
    assert "environment" not in JOBS["release-pr"]
    unpublished = "needs.sdk-check.outputs.publish == 'true'"
    assert JOBS["sdk-build"]["if"] == JOBS["sdk-publish"]["if"] == unpublished


def test_bump_targets_main_with_uv():
    script = "\n".join(step.get("run", "") for step in _steps("release-pr"))
    assert "uv version --bump patch --frozen" in script
    assert "git add pyproject.toml uv.lock" in script
    assert script.count("--base ") == script.count("--base main") == 2
    assert "--force" not in script


def test_no_merge_can_cancel_another_merges_template():
    assert "concurrency" not in WORKFLOW
    concurrency = JOBS["template-publish"]["concurrency"]
    assert "github.sha" in concurrency["group"]
    assert concurrency["cancel-in-progress"] is False


def test_deployment_pr_only_after_certification():
    assert _needs("deployment-bump") == {"template-publish"}
    assert JOBS["deployment-bump"]["if"] == (
        "${{ !cancelled() && needs.template-publish.result == 'success' }}"
    )
    assert "needs.template-publish.outputs.release" in _dump("deployment-bump")
    assert JOBS["template-publish"]["outputs"] == {
        "release": "${{ steps.release.outputs.json }}"
    }
    export = next(
        step for step in _steps("template-publish") if step.get("id") == "release"
    )
    names = [step.get("name") for step in _steps("template-publish")]
    assert names.index("Build, publish, and certify") < names.index(export["name"])
    # Inside `echo "...$(jq ...)"` a failing jq would still exit 0 and export nothing.
    assert (
        export["run"].splitlines()[0]
        == 'json="$(jq -c . dist/e2b-template-release.json)"'
    )


def test_deployment_pr_targets_main_in_both_regions():
    checkouts = [
        step["with"]
        for step in _steps("deployment-bump")
        if step.get("uses", "").startswith("actions/checkout@")
        and "repository" in step.get("with", {})
    ]
    assert [(c["repository"], c["ref"]) for c in checkouts] == [
        ("future-agi/deployment", "main")
    ]
    script = "\n".join(step.get("run", "") for step in _steps("deployment-bump"))
    assert script.count("--base ") == script.count("--base main") == 2
    assert "deployment/us/gcp/deployment/values.yaml" in script
    assert "deployment/eu/gcp/deployment/values.yaml" in script
    assert (
        "git add us/gcp/deployment/values.yaml eu/gcp/deployment/values.yaml" in script
    )
    assert "--repo future-agi/deployment" in script


def test_deployment_job_never_prints_a_values_file():
    # This repository is public, so anything a job prints is world-readable.
    script = "\n".join(step.get("run", "") for step in _steps("deployment-bump"))
    diffs = [match.strip() for match in re.findall(r"git diff[^\n;&|]*", script)]
    assert diffs == ["git diff --cached --quiet"]
    assert not re.search(r"set\s+[-+]\w*x", script)
    for printer in ("git show", "git log", "git stash", "xtrace"):
        assert printer not in script, printer
    assert not re.search(r"(?<![-\w])(cat|head|tail|less|grep|sed|awk)\b", script)
    patches = [line for line in script.splitlines() if "-X PATCH" in line]
    assert patches and all("--silent" in line for line in patches)


def test_secrets_stay_in_their_job():
    assert WORKFLOW["permissions"] == {"contents": "read"}
    for job in JOBS:
        dumped = _dump(job)
        assert ("RELEASE_BOT_" in dumped) == (
            job in {"deployment-bump", "release-pr"}
        ), job
        assert ("E2B_API_KEY" in dumped) == (job == "template-publish"), job
        assert ("id-token" in dumped) == (job == "sdk-publish"), job
        assert ("packages" in JOBS[job].get("permissions", {})) == (
            job == "template-publish"
        ), job


def test_each_bot_token_reaches_one_repository():
    for job, repository in (
        ("release-pr", "agent-learning-kit"),
        ("deployment-bump", "deployment"),
    ):
        minted = [
            step["with"]
            for step in _steps(job)
            if step.get("uses", "").startswith("actions/create-github-app-token@")
        ]
        assert len(minted) == 1, job
        token = minted[0]
        assert token["owner"] == "future-agi", job
        assert token["repositories"] == repository, job
        permissions = {k: v for k, v in token.items() if k.startswith("permission-")}
        assert permissions == {
            "permission-contents": "write",
            "permission-pull-requests": "write",
        }, job


def test_actions_are_pinned():
    used = [
        step["uses"] for job in JOBS.values() for step in job["steps"] if "uses" in step
    ]
    assert used
    for action in used:
        assert re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", action), action


def test_no_empty_expression():
    assert not re.search(r"\$\{\{\s*\}\}", TEXT)


def test_bump_job_is_wired():
    job = JOBS["release-pr"]
    token = "${{ steps.app-token.outputs.token }}"
    (checkout,) = [
        s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout@")
    ]
    (setup_uv,) = [
        s for s in job["steps"] if s.get("uses", "").startswith("astral-sh/setup-uv@")
    ]
    (bump,) = [
        s for s in job["steps"] if s.get("name") == "Open the version-bump pull request"
    ]
    assert checkout["with"] == {"token": token}
    assert setup_uv["with"]["version"] == "${{ env.UV_VERSION }}"
    assert bump["env"] == {"GH_TOKEN": token}
    assert bump["run"].startswith("set -euo pipefail\n")
    assert job["concurrency"] == {"group": "release-pr", "cancel-in-progress": False}


def test_ci_runs_the_release_tests():
    smoke = (WORKFLOWS / "sdk-smoke.yml").read_text(encoding="utf-8")
    for name in (
        "test_bump_deployment_values",
        "test_release_workflow",
        "test_release_pr_step",
    ):
        assert f"tests/{name}.py" in smoke, name
