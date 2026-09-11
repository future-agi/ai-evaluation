import asyncio
from io import BytesIO
import json
import os

from fastapi import UploadFile
from fastapi.testclient import TestClient
import pytest

from fi.alk.harness.sandbox_server import (
    LocalSandbox,
    LocalSandboxRequest,
    SandboxAdjustmentRequest,
    SandboxRerunRequest,
    _CONTROLLER_TOKEN,
    _presentation_value,
    _process_identity,
    _worker_failure_retryable,
    create_app,
)
from fi.alk.harness.secrets import resolve_worker_secrets


def test_presentation_redaction_handles_prose_with_email_before_url():
    prose = "Contact owner@example.com or see https://example.com/docs"

    assert _presentation_value(prose) == prose
    assert (
        _presentation_value("postgres://user:password@database:5432/app")
        == "postgres://[redacted]@database:5432/app"
    )


def test_local_sandbox_rejects_source_outside_allowed_root(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv("ALK_SANDBOX_SOURCE_ROOTS", str(allowed))
    client = TestClient(create_app(tmp_path / "state"))

    response = client.post("/v1/jobs", json={"source_path": str(outside)})

    assert response.status_code == 403


def test_second_sandbox_instance_does_not_orphan_live_controller(tmp_path):
    root = tmp_path / "state"
    first = LocalSandbox(root)
    job = first.jobs_root / "live-job"
    job.mkdir()
    state_path = job / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "stage": "running",
                "controller_pid": os.getpid(),
                "controller_identity": _process_identity(os.getpid()),
                "controller_token": _CONTROLLER_TOKEN,
            }
        ),
        encoding="utf-8",
    )

    LocalSandbox(root)

    assert json.loads(state_path.read_text())["stage"] == "running"


def test_local_sandbox_persists_and_lists_job_without_secrets(tmp_path, monkeypatch):
    source = tmp_path / "sources" / "agent"
    source.mkdir(parents=True)
    monkeypatch.setenv("ALK_SANDBOX_SOURCE_ROOTS", str(tmp_path / "sources"))
    monkeypatch.setenv("ALK_SANDBOX_TOKEN", "test-token")
    client = TestClient(create_app(tmp_path / "state"))

    response = client.post(
        "/v1/jobs",
        headers={"Authorization": "Bearer test-token"},
        json={
            "source_path": str(source),
            "scenario_count": 7,
            "secret_refs": {
                "livekit": {
                    "manager": "environment",
                    "key": "LK_URL",
                    "purpose": "livekit_url",
                }
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"]["total_scenarios"] == 7
    assert body["job"]["agent"]["secret_refs"]["livekit"]["key"] == "LK_URL"
    assert (
        client.get("/v1/jobs", headers={"Authorization": "Bearer test-token"}).json()[
            0
        ]["job"]["job_id"]
        == body["job"]["job_id"]
    )


def test_local_sandbox_reports_live_stage_timestamp_and_detail_from_events(
    tmp_path, monkeypatch
):
    source = tmp_path / "sources" / "agent"
    source.mkdir(parents=True)
    monkeypatch.setenv("ALK_SANDBOX_SOURCE_ROOTS", str(tmp_path / "sources"))
    state_root = tmp_path / "state"
    client = TestClient(create_app(state_root))
    submitted = client.post("/v1/jobs", json={"source_path": str(source)}).json()
    job_id = submitted["job"]["job_id"]
    run_id = submitted["job"]["run_id"]
    event_time = "2026-08-22T01:02:03+00:00"
    events_path = state_root / "artifacts" / run_id / "harness-events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text(
        json.dumps(
            {
                "type": "harness.stage.started",
                "wall_time": event_time,
                "payload": {"stage": "scenarios"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    current = client.get(f"/v1/jobs/{job_id}").json()["status"]

    assert current["stage"] == "generating_scenarios"
    assert current["updated_at"] == "2026-08-22T01:02:03Z"
    assert current["detail"] == "generating and validating scenarios"


def test_local_sandbox_exposes_generated_stage_outputs_and_adjustments(
    tmp_path, monkeypatch
):
    source = tmp_path / "sources" / "agent"
    source.mkdir(parents=True)
    monkeypatch.setenv("ALK_SANDBOX_SOURCE_ROOTS", str(tmp_path / "sources"))
    sandbox = LocalSandbox(tmp_path / "state")

    async def idle(_job, _source):
        await asyncio.sleep(0)

    sandbox._execute = idle

    async def submit():
        response = sandbox.submit(LocalSandboxRequest(source_path=str(source)))
        await sandbox._tasks[response.job.job_id]
        return response

    response = asyncio.run(submit())
    output = tmp_path / "state" / "artifacts" / response.job.run_id
    output.mkdir(parents=True)
    (output / "contract.json").write_text(
        json.dumps(
            {
                "agent": "agent",
                "one_liner": "Books appointments",
                "tools": [{"name": "book"}],
                "hard_constraints": ["confirm first"],
                "real_use_cases": ["new booking"],
            }
        ),
        encoding="utf-8",
    )
    (output / "environment.json").write_text(
        json.dumps(
            {
                "source": "/private/source",
                "compose_file": "/private/docker-compose.yml",
                "compose_override_file": "/private/override.yml",
                "services": ["postgres"],
                "overrides": {"DATABASE_URL": "postgres://test"},
                "running": True,
            }
        ),
        encoding="utf-8",
    )
    (output / "scenarios.json").write_text(
        json.dumps([{"name": "booking", "instruction": "Book tomorrow"}]),
        encoding="utf-8",
    )

    adjusted = sandbox.adjust(
        response.job.job_id,
        SandboxAdjustmentRequest(
            instruction="Add 10 more scenarios covering payment failures"
        ),
    )

    assert [item["kind"] for item in adjusted.stage_outputs] == [
        "contract",
        "environment",
        "scenarios",
    ]
    assert "source" not in adjusted.stage_outputs[1]["data"]
    assert "compose_override_file" not in adjusted.stage_outputs[1]["data"]
    assert adjusted.status.total_scenarios == 10
    assert adjusted.adjustments[0]["target_stage"] == "scenarios"
    assert adjusted.adjustments[0]["scenario_delta"] == 10
    assert adjusted.adjustments[0]["status"] == "pending"


def test_local_sandbox_reports_committed_scenario_progress_while_worker_runs(
    tmp_path, monkeypatch
):
    source = tmp_path / "sources" / "agent"
    source.mkdir(parents=True)
    monkeypatch.setenv("ALK_SANDBOX_SOURCE_ROOTS", str(tmp_path / "sources"))
    state_root = tmp_path / "state"
    client = TestClient(create_app(state_root))
    submitted = client.post(
        "/v1/jobs", json={"source_path": str(source), "scenario_count": 3}
    ).json()
    job_id = submitted["job"]["job_id"]
    run_id = submitted["job"]["run_id"]
    events_path = state_root / "artifacts" / run_id / "harness-events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text(
        json.dumps(
            {
                "type": "harness.stage.started",
                "wall_time": "2026-08-22T01:02:03+00:00",
                "payload": {"stage": "running"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    campaign = state_root / "artifacts" / run_id / "runs" / "run-20260822-010203"
    (campaign / "scenario-one").mkdir(parents=True)
    (campaign / "scenario-one" / "result.json").write_text("{}", encoding="utf-8")
    # Nested results and unfinished scenario directories must not inflate progress.
    (campaign / "scenario-one" / "sdk").mkdir()
    (campaign / "scenario-one" / "sdk" / "result.json").write_text(
        "{}", encoding="utf-8"
    )
    (campaign / "scenario-two").mkdir()

    current = client.get(f"/v1/jobs/{job_id}").json()["status"]

    assert current["stage"] == "running"
    assert current["completed_scenarios"] == 1
    assert current["total_scenarios"] == 3


def test_local_sandbox_exposes_secret_safe_live_stage_outputs(tmp_path, monkeypatch):
    source = tmp_path / "sources" / "agent"
    source.mkdir(parents=True)
    monkeypatch.setenv("ALK_SANDBOX_SOURCE_ROOTS", str(tmp_path / "sources"))
    state_root = tmp_path / "state"
    sandbox = LocalSandbox(state_root)

    async def no_execution(_job, _source):
        return None

    sandbox._execute = no_execution

    async def submit():
        response = sandbox.submit(LocalSandboxRequest(source_path=str(source)))
        await sandbox._tasks[response.job.job_id]
        return response

    submitted = asyncio.run(submit())
    run_root = sandbox.artifacts_root / submitted.job.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "contract.json").write_text(
        json.dumps(
            {
                "one_liner": "Books rides",
                "modality": "voice",
                "tools": [{"name": "book_ride", "args": ["secret"]}],
                "hard_constraints": ["Confirm before booking"],
            }
        ),
        encoding="utf-8",
    )
    (run_root / "environment.json").write_text(
        json.dumps(
            {
                "project": "isolated",
                "services": ["postgres"],
                "overrides": {
                    "DATABASE_URL": "postgresql://user:password@postgres:5432/app"
                },
            }
        ),
        encoding="utf-8",
    )
    (run_root / "scenarios.json").write_text(
        json.dumps(
            [
                {
                    "name": "booking",
                    "instruction": "Book a ride",
                    "use_case": "happy path",
                    "private_reference": "must-not-be-returned",
                }
            ]
        ),
        encoding="utf-8",
    )
    (run_root / "platform.json").write_text(
        json.dumps(
            {
                "run_test_id": "11111111-1111-1111-1111-111111111111",
                "test_execution_id": "22222222-2222-2222-2222-222222222222",
                "url": "https://untrusted.example/ignored",
            }
        ),
        encoding="utf-8",
    )

    response = sandbox.get(submitted.job.job_id).model_dump(mode="json")
    serialized = json.dumps(response["stage_outputs"])

    assert [item["kind"] for item in response["stage_outputs"]] == [
        "contract",
        "environment",
        "scenarios",
        "simulation",
    ]
    assert "book_ride" in serialized
    assert "postgresql://***@postgres:5432/app" in serialized
    assert "password" not in serialized
    assert "must-not-be-returned" not in serialized
    assert (
        "/dashboard/simulate/test/11111111-1111-1111-1111-111111111111/"
        "22222222-2222-2222-2222-222222222222"
    ) in serialized
    assert "untrusted.example" not in serialized


def test_preflight_discovers_connectors_and_missing_credentials_without_values(
    tmp_path, monkeypatch
):
    source = tmp_path / "sources" / "agent"
    source.mkdir(parents=True)
    (source / "worker.py").write_text(
        "from livekit import agents\n"
        "import os\n"
        "key = os.environ['LIVEKIT_API_KEY']\n"
        "database = os.environ['DATABASE_URL']\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ALK_SANDBOX_SOURCE_ROOTS", str(tmp_path / "sources"))
    client = TestClient(create_app(tmp_path / "state"))

    response = client.post("/v1/preflight", json={"source_path": str(source)})

    assert response.status_code == 200
    body = response.json()
    assert body["source_kind"] == "local_repository"
    assert body["ready_to_submit"] is False
    assert "livekit" in body["credentials"]["detected_connectors"]
    requirements = {
        item["environment_name"]: item for item in body["credentials"]["requirements"]
    }
    assert requirements["LIVEKIT_API_KEY"]["status"] == "missing"
    assert requirements["DATABASE_URL"]["status"] == "harness_provided"
    assert "value" not in json.dumps(body).lower()


def test_preflight_accepts_non_secret_connector_configuration(tmp_path, monkeypatch):
    source = tmp_path / "sources" / "agent"
    source.mkdir(parents=True)
    (source / "worker.py").write_text(
        'import os\nagent = os.environ["REMOTE_AGENT_ID"]\n', encoding="utf-8"
    )
    monkeypatch.setenv("ALK_SANDBOX_SOURCE_ROOTS", str(tmp_path / "sources"))
    client = TestClient(create_app(tmp_path / "state"))

    response = client.post(
        "/v1/preflight",
        json={
            "source_path": str(source),
            "connector_config": {"REMOTE_AGENT_ID": "agent-42"},
        },
    )

    assert response.status_code == 200
    assert response.json()["ready_to_submit"] is True
    requirement = response.json()["credentials"]["requirements"][0]
    assert requirement["status"] == "configured"
    assert "agent-42" not in response.text


def test_preflight_accepts_uploaded_environment_without_returning_values(
    tmp_path, monkeypatch
):
    source = tmp_path / "sources" / "agent"
    source.mkdir(parents=True)
    (source / "worker.py").write_text(
        'import os\nkey = os.environ["OPENAI_API_KEY"]\n', encoding="utf-8"
    )
    monkeypatch.setenv("ALK_SANDBOX_SOURCE_ROOTS", str(tmp_path / "sources"))
    client = TestClient(create_app(tmp_path / "state"))

    response = client.post(
        "/v1/preflight",
        json={
            "source_path": str(source),
            "environment_values": {"OPENAI_API_KEY": "never-persist-this"},
        },
    )

    assert response.status_code == 200
    assert response.json()["credentials"]["requirements"][0]["status"] == "configured"
    assert "never-persist-this" not in response.text


def test_uploaded_environment_is_mounted_for_worker_but_not_persisted(
    tmp_path, monkeypatch
):
    source = tmp_path / "sources" / "agent"
    source.mkdir(parents=True)
    monkeypatch.setenv("ALK_SANDBOX_SOURCE_ROOTS", str(tmp_path / "sources"))
    sandbox = LocalSandbox(tmp_path / "state")
    observed = {}

    async def capture(job, _source):
        observed.update(
            resolve_worker_secrets(
                job.agent.secret_refs,
                environment={
                    **os.environ,
                    **sandbox._ephemeral_secrets[job.job_id],
                },
            )
        )
        sandbox._ephemeral_secrets.pop(job.job_id, None)

    sandbox._execute = capture

    async def submit():
        response = sandbox.submit(
            LocalSandboxRequest(
                source_path=str(source),
                environment_values={"OPENAI_API_KEY": "never-persist-this"},
            )
        )
        await sandbox._tasks[response.job.job_id]
        return response

    response = asyncio.run(submit())
    persisted = (
        tmp_path / "state" / "jobs" / response.job.job_id / "job.json"
    ).read_text()

    assert observed == {"OPENAI_API_KEY": "never-persist-this"}
    assert "never-persist-this" not in persisted
    reference = response.job.agent.secret_refs["OPENAI_API_KEY"]
    assert reference.manager == "mounted"
    assert reference.key.startswith("ALK_JOB_")


def test_saved_session_rerun_reuses_artifacts_and_keeps_fresh_values_ephemeral(
    tmp_path, monkeypatch
):
    source = tmp_path / "sources" / "agent"
    source.mkdir(parents=True)
    monkeypatch.setenv("ALK_SANDBOX_SOURCE_ROOTS", str(tmp_path / "sources"))
    sandbox = LocalSandbox(tmp_path / "state")

    async def no_initial_execution(job, _source):
        sandbox._ephemeral_secrets.pop(job.job_id, None)

    sandbox._execute = no_initial_execution

    async def exercise():
        response = sandbox.submit(LocalSandboxRequest(source_path=str(source)))
        await sandbox._tasks[response.job.job_id]
        job_id = response.job.job_id
        run_root = sandbox.artifacts_root / response.job.run_id
        (run_root / "environment-bundle").mkdir(parents=True)
        for path in (
            run_root / "contract.json",
            run_root / "scenarios.json",
            run_root / "environment-bundle" / "manifest.json",
            run_root / "platform.json",
        ):
            path.write_text("{}", encoding="utf-8")
        state_path = sandbox.jobs_root / job_id / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["stage"] = "completed"
        state_path.write_text(json.dumps(state), encoding="utf-8")

        observed = {}

        async def capture(job, only):
            observed["job"] = job.job_id
            observed["only"] = only
            observed["values"] = dict(sandbox._ephemeral_secrets[job.job_id])
            sandbox._ephemeral_secrets.pop(job.job_id, None)

        sandbox._execute_rerun = capture
        queued = sandbox.rerun(
            job_id,
            SandboxRerunRequest(
                environment_values={"OPENAI_API_KEY": "fresh-never-persist"}
            ),
        )
        await sandbox._tasks[job_id]
        return queued, observed, state_path

    queued, observed, state_path = asyncio.run(exercise())

    assert queued.status.stage.value == "queued"
    assert observed == {
        "job": queued.job.job_id,
        "only": [],
        "values": {"OPENAI_API_KEY": "fresh-never-persist"},
    }
    assert "fresh-never-persist" not in state_path.read_text(encoding="utf-8")
    assert "fresh-never-persist" not in (state_path.parent / "job.json").read_text(
        encoding="utf-8"
    )


def test_uploaded_environment_rejects_runner_control_and_reference_conflicts(
    tmp_path, monkeypatch
):
    source = tmp_path / "sources" / "agent"
    source.mkdir(parents=True)
    monkeypatch.setenv("ALK_SANDBOX_SOURCE_ROOTS", str(tmp_path / "sources"))
    client = TestClient(create_app(tmp_path / "state"))

    reserved = client.post(
        "/v1/jobs",
        json={
            "source_path": str(source),
            "environment_values": {"DOCKER_HOST": "host-escape"},
        },
    )
    conflict = client.post(
        "/v1/jobs",
        json={
            "source_path": str(source),
            "environment_values": {"OPENAI_API_KEY": "raw"},
            "secret_refs": {
                "OPENAI_API_KEY": {
                    "manager": "environment",
                    "key": "OPENAI_API_KEY",
                    "purpose": "existing reference",
                }
            },
        },
    )

    assert reserved.status_code == 422
    assert conflict.status_code == 422


def test_uploaded_folder_becomes_sandbox_owned_source(tmp_path):
    client = TestClient(create_app(tmp_path / "state"))

    uploaded = client.post(
        "/v1/sources",
        files=[
            ("files", ("agent.py", b"print('ready')\n", "text/x-python")),
            ("files", ("requirements.txt", b"fastapi\n", "text/plain")),
            ("paths", (None, "agent.py")),
            ("paths", (None, "requirements.txt")),
            ("name", (None, "customer-agent")),
        ],
    )

    assert uploaded.status_code == 201
    body = uploaded.json()
    assert body["name"] == "customer-agent"
    assert body["file_count"] == 2
    source_id = body["source_id"]
    source = tmp_path / "state" / "uploads" / source_id
    assert source.joinpath("agent.py").read_text() == "print('ready')\n"

    preflight = client.post("/v1/preflight", json={"source_id": source_id})
    assert preflight.status_code == 200
    assert preflight.json()["source_kind"] == "archive"
    assert preflight.json()["source_label"] == "customer-agent"


def test_uploaded_folder_rejects_traversal_and_embedded_env(tmp_path):
    client = TestClient(create_app(tmp_path / "state"))

    traversal = client.post(
        "/v1/sources",
        files=[
            ("files", ("escape.py", b"bad", "text/plain")),
            ("paths", (None, "../escape.py")),
        ],
    )
    secret_file = client.post(
        "/v1/sources",
        files=[
            ("files", (".env", b"API_KEY=secret", "text/plain")),
            ("paths", (None, ".env")),
        ],
    )

    assert traversal.status_code == 400
    assert secret_file.status_code == 400
    assert not tmp_path.joinpath("escape.py").exists()


def test_secret_file_upload_is_opaque_one_time_and_job_scoped(tmp_path):
    sandbox = LocalSandbox(tmp_path / "state")
    raw = b'{"type":"service_account","project_id":"customer"}'
    uploaded = asyncio.run(
        sandbox.upload_secret_file(
            UploadFile(filename="customer.json", file=BytesIO(raw)),
            "GOOGLE_APPLICATION_CREDENTIALS",
        )
    )

    assert uploaded.size == len(raw)
    assert uploaded.secret_ref.manager == "mounted"
    assert raw.decode() not in uploaded.model_dump_json()
    with pytest.raises(Exception, match="reference_invalid"):
        sandbox._claim_secret_files(
            "wrong-alias", {"AWS_SHARED_CREDENTIALS_FILE": uploaded.secret_ref}
        )
    claimed, names = sandbox._claim_secret_files(
        "job-one", {uploaded.environment_name: uploaded.secret_ref}
    )
    claimed_path = next(iter(claimed.values()))
    assert names == {"GOOGLE_APPLICATION_CREDENTIALS"}
    assert open(claimed_path, "rb").read() == raw
    assert oct(os.stat(claimed_path).st_mode & 0o777) == "0o400"
    with pytest.raises(Exception, match="unavailable_or_consumed"):
        sandbox._claim_secret_files(
            "job-two", {uploaded.environment_name: uploaded.secret_ref}
        )

    sandbox._delete_job_secret_files("job-one")
    assert not os.path.exists(claimed_path)


def test_secret_file_endpoint_rejects_invalid_google_json(tmp_path):
    client = TestClient(create_app(tmp_path / "state"))

    response = client.post(
        "/v1/secret-files",
        files={"file": ("credentials.json", b"not-json", "application/json")},
        data={"environment_name": "GOOGLE_APPLICATION_CREDENTIALS"},
    )

    assert response.status_code == 400
    assert "valid JSON object" in response.json()["detail"]


def test_preflight_does_not_call_infrastructure_only_compose_execution_ready(
    tmp_path, monkeypatch
):
    source = tmp_path / "sources" / "agent"
    source.mkdir(parents=True)
    (source / "compose.yml").write_text(
        "services:\n  postgres:\n    image: postgres:17\n  redis:\n    image: redis:7\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ALK_SANDBOX_SOURCE_ROOTS", str(tmp_path / "sources"))
    client = TestClient(create_app(tmp_path / "state"))

    response = client.post("/v1/preflight", json={"source_path": str(source)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["packaging"]["ready"] is True
    assert payload["packaging"]["agent_runtime_packaged"] is False
    assert payload["ready_to_submit"] is False


def test_preflight_accepts_public_github_url_and_requires_app_for_private_repo(
    tmp_path,
):
    client = TestClient(create_app(tmp_path / "state"))

    public = client.post(
        "/v1/preflight",
        json={"github_repository": "https://github.com/org/agent.git"},
    )
    private = client.post(
        "/v1/preflight",
        json={
            "github_repository": "org/private-agent",
            "github_visibility": "private",
        },
    )

    assert public.status_code == 200
    assert public.json()["source_label"] == "org/agent"
    assert public.json()["ready_to_submit"] is True
    assert public.json()["checkout_required"] is True
    assert private.json()["ready_to_submit"] is False
    requirement = private.json()["credentials"]["requirements"][0]
    assert requirement["accepted_secret_types"] == ["github_app_installation"]


def test_public_github_branch_url_populates_job_ref(tmp_path):
    client = TestClient(create_app(tmp_path / "state"))

    response = client.post(
        "/v1/jobs",
        json={
            "github_repository": "https://github.com/acme/agent/tree/feat/harness",
        },
    )

    assert response.status_code == 200
    source = response.json()["job"]["source"]
    assert source["repository"] == "acme/agent"
    assert source["ref"] == "feat/harness"


def test_private_github_submission_becomes_hosted_job(tmp_path):
    client = TestClient(create_app(tmp_path / "state"))

    response = client.post(
        "/v1/jobs",
        json={
            "github_repository": "customer/private-agent",
            "github_visibility": "private",
            "github_installation_id": "installation-42",
        },
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["execution"] == "hosted"
    assert job["source"]["installation_id"] == "installation-42"
    assert "token" not in json.dumps(job).lower()


def test_job_submission_requires_exactly_one_source(tmp_path):
    client = TestClient(create_app(tmp_path / "state"))

    none = client.post("/v1/jobs", json={})
    both = client.post(
        "/v1/jobs",
        json={"source_path": str(tmp_path), "github_repository": "org/agent"},
    )

    assert none.status_code == 422
    assert both.status_code == 422


def test_retry_policy_never_retries_agent_or_grading_failures():
    allowed = ["infrastructure", "connectivity"]

    assert _worker_failure_retryable(
        1, {"domain": "infrastructure", "retryable": True}, allowed
    )
    assert not _worker_failure_retryable(
        1, {"domain": "agent", "retryable": True}, allowed
    )
    assert not _worker_failure_retryable(
        1, {"domain": "grading", "retryable": True}, allowed
    )
    assert not _worker_failure_retryable(
        0, {"domain": "infrastructure", "retryable": True}, allowed
    )
