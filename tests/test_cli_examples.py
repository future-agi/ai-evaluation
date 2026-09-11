from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from fi.alk import actions, trinity
from fi.alk.cli import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = PROJECT_ROOT / "examples"


def _load_example_module(name: str):
    path = EXAMPLES / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sdk_openenv_environment_simulation_example_runs(tmp_path):
    module = _load_example_module("sdk_openenv_environment_simulation.py")

    manifest = module.build_manifest()
    assert manifest["name"] == "sdk-openenv-environment-simulation"
    assert [env["type"] for env in manifest["simulation"]["environments"]] == [
        "openenv"
    ]
    assert manifest["evaluation"]["agent_report"]["config"]["openenv_quality"][
        "min_step_count"
    ] == 2

    output_path = tmp_path / "sdk-openenv-environment-result.json"
    result = module.run(output_path)

    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["status"] == "passed"
    assert result["status"] == "passed"
    assert result["summary"]["metric_averages"]["openenv_coverage"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"]["openenv_quality"] == (
        pytest.approx(1.0)
    )
    state = result["report"]["results"][0]["metadata"]["environment_state"]["openenv"]
    summary = state["summary"]
    assert summary["reset_count"] == 1
    assert summary["step_count"] == 2
    assert summary["done"] is True
    assert summary["failure_count"] == 1
    assert summary["sandbox_enabled"] is True
    assert summary["requires_external_service"] is False
    assert result["openenv_environment_manifest"]["simulation"]["environments"][0][
        "type"
    ] == "openenv"


def test_sdk_openenv_environment_optimization_example_runs(tmp_path):
    module = _load_example_module("sdk_openenv_environment_optimization.py")

    manifest = module.build_manifest(required_env=())
    assert manifest["name"] == "sdk-openenv-environment-optimization"
    assert manifest["required_env"] == []
    assert manifest["optimization"]["scoring"]["layers"] == ["openenv"]
    candidates = manifest["optimization"]["target"]["search_space"][
        "simulation.environments"
    ]
    assert [
        candidate[0]["data"]["metadata"]["candidate_profile"]
        for candidate in candidates
    ] == [
        "weak_openenv_reset_step_only",
        "partial_openenv_no_failure_injection",
        "verified_openenv_replay",
    ]

    output_path = tmp_path / "sdk-openenv-environment-optimization.json"
    result = module.run(output_path, required_env=())
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.optimization.v1"
    assert result["status"] == "passed"
    assert result["summary"]["optimization_score"] == pytest.approx(1.0)
    assert result["summary"]["evaluation_score"] == pytest.approx(1.0)
    assert result["summary"]["candidate_lineage_count"] == 3
    best_config = result["optimization"]["best_config"]
    best_environment = best_config["simulation"]["environments"][0]
    assert best_environment["type"] == "openenv"
    assert best_environment["data"]["metadata"]["candidate_profile"] == (
        "verified_openenv_replay"
    )
    best_history = max(
        result["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["patch"].keys() == {"simulation.environments"}
    assert best_history["score"] == pytest.approx(1.0)
    assert best_history["metrics"]["openenv_coverage"] == pytest.approx(1.0)
    assert best_history["metrics"]["openenv_quality"] == pytest.approx(1.0)


def test_sdk_framework_adapter_openenv_trace_example_runs(tmp_path):
    module = _load_example_module("sdk_framework_adapter_openenv_trace.py")

    output_path = tmp_path / "sdk-framework-adapter-openenv-trace.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    manifest = result["framework_adapter_openenv_trace_manifest"]
    assert manifest["agent"]["framework"] == "openenv"
    assert manifest["agent"]["method"] == "run"
    config = manifest["evaluation"]["agent_report"]["config"]
    runtime_contract = config["framework_runtime_contract"]
    assert runtime_contract["required_state_keys"] == ["openenv"]
    assert set(runtime_contract["required_signals"]) >= {
        "artifact",
        "event",
        "openenv",
        "state",
    }
    assert set(config["required_openenv"]) >= {
        "openenv",
        "state",
        "observation",
        "reset",
        "step",
        "action",
        "reward",
        "done",
        "terminated",
        "metadata",
        "sandbox",
        "failure_injection",
        "in_process",
        "local",
    }
    openenv_quality = config["openenv_quality"]
    assert openenv_quality["min_reset_count"] == 1
    assert openenv_quality["min_step_count"] == 2
    assert openenv_quality["min_action_route_count"] == 2
    assert openenv_quality["min_failure_count"] == 1
    assert openenv_quality["min_reward_total"] == pytest.approx(1.0)
    assert openenv_quality["max_error_count"] == 0
    assert openenv_quality["require_done"] is True
    assert openenv_quality["require_terminated"] is True
    assert openenv_quality["require_sandbox"] is True
    assert openenv_quality["require_metadata_capture"] is True
    assert openenv_quality["require_no_external_service"] is True
    assert openenv_quality["require_deterministic_reset"] is True
    assert openenv_quality["required_runtime"] == "in_process"
    assert openenv_quality["required_transport"] == "local"
    assert openenv_quality["required_isolation"] == "process"
    assert config["metric_weights"]["openenv_coverage"] == pytest.approx(4.0)
    assert config["metric_weights"]["openenv_quality"] == pytest.approx(4.0)
    assert result["summary"]["metric_averages"]["openenv_coverage"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"]["openenv_quality"] == (
        pytest.approx(1.0)
    )
    state = result["report"]["results"][0]["metadata"]["environment_state"]
    openenv = state["openenv"]
    summary = openenv["summary"]
    assert summary["reset_count"] == 1
    assert summary["step_count"] == 2
    assert summary["done"] is True
    assert summary["failure_count"] == 1
    assert summary["sandbox_enabled"] is True
    assert summary["requires_external_service"] is False
    output = state["framework_runtime"]["invocations"][0]["output"]
    assert "openenv" in output["state_keys"]
    assert {"trace"} <= set(output["artifact_types"])
    assert {"openenv"} <= set(output["event_types"])
    assert output["openenv_summary"]["step_count"] == 2


def test_framework_openenv_manifest_runs_through_cli(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "AGENT_LEARNING_OPENENV_EXAMPLE_KEY",
        "real-local-openenv-framework-key",
    )
    output_path = tmp_path / "framework-openenv.json"

    exit_code = main([
        "run",
        str(EXAMPLES / "framework_openenv_manifest.json"),
        "--output",
        str(output_path),
    ])

    assert exit_code == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    metrics = result["summary"]["metric_averages"]
    assert metrics["framework_runtime_contract"] == pytest.approx(1.0)
    assert metrics["openenv_coverage"] == pytest.approx(1.0)
    assert metrics["openenv_quality"] == pytest.approx(1.0)
    state = result["report"]["results"][0]["metadata"]["environment_state"]
    assert state["openenv"]["summary"]["step_count"] == 2
    assert state["framework_runtime"]["invocations"][0]["output"][
        "openenv_summary"
    ]["done"] is True


@pytest.mark.parametrize(
    ("command", "example", "kind", "required_env"),
    [
        (
            "run",
            "run_manifest.json",
            "agent-learning.run.v1",
            "AGENT_LEARNING_RUN_EXAMPLE_KEY",
        ),
        ("eval", "eval_suite.json", "agent-learning.eval.v1", None),
        ("eval", "artifact_task_eval_suite.json", "agent-learning.eval.v1", None),
        (
            "eval-artifact",
            "fixtures/task_artifacts/refund_task_run.json",
            "agent-learning.artifact-evaluation.v1",
            None,
        ),
        (
            "eval-task",
            "task_evidence.json",
            "agent-learning.artifact-evaluation.v1",
            None,
        ),
        (
            "redteam",
            "redteam_manifest.json",
            "agent-learning.redteam.v1",
            "AGENT_LEARNING_REDTEAM_EXAMPLE_KEY",
        ),
        (
            "redteam",
            "long_horizon_redteam_manifest.json",
            "agent-learning.redteam.v1",
            "AGENT_LEARNING_LONG_HORIZON_REDTEAM_KEY",
        ),
        (
            "optimize",
            "optimization_manifest.json",
            "agent-learning.optimization.v1",
            "AGENT_LEARNING_OPTIMIZE_EXAMPLE_KEY",
        ),
        (
            "optimize",
            "long_horizon_redteam_optimization.json",
            "agent-learning.optimization.v1",
            "AGENT_LEARNING_LONG_HORIZON_REDTEAM_OPT_EXAMPLE_KEY",
        ),
        (
            "optimize",
            "redteam_society_optimization.json",
            "agent-learning.optimization.v1",
            "AGENT_LEARNING_REDTEAM_SOCIETY_OPT_EXAMPLE_KEY",
        ),
        (
            "optimize",
            "redteam_causal_attribution_optimization.json",
            "agent-learning.optimization.v1",
            "AGENT_LEARNING_REDTEAM_CAUSAL_ATTRIBUTION_OPT_EXAMPLE_KEY",
        ),
        (
            "optimize",
            "report_repair_optimization.json",
            "agent-learning.optimization.v1",
            "AGENT_LEARNING_REPORT_REPAIR_OPT_EXAMPLE_KEY",
        ),
        (
            "optimize",
            "framework_import_repair_optimization.json",
            "agent-learning.optimization.v1",
            "AGENT_LEARNING_FRAMEWORK_IMPORT_REPAIR_OPT_EXAMPLE_KEY",
        ),
        (
            "optimize-eval",
            "eval_suite_optimization.json",
            "agent-learning.eval-optimization.v1",
            None,
        ),
        (
            "optimize-suite",
            "suite_optimization.json",
            "agent-learning.suite-optimization.v1",
            [
                "AGENT_LEARNING_SUITE_OPT_EXAMPLE_KEY",
                "AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY",
            ],
        ),
        (
            "suite",
            "agent_learning_suite.json",
            "agent-learning.suite.v1",
            [
                "AGENT_LEARNING_RUN_EXAMPLE_KEY",
                "AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY",
                "AGENT_LEARNING_CUSTOM_FRAMEWORK_OPT_EXAMPLE_KEY",
                "AGENT_LEARNING_SOCIAL_MEMORY_OPT_EXAMPLE_KEY",
                "AGENT_LEARNING_REDTEAM_EXAMPLE_KEY",
                "AGENT_LEARNING_WORLD_FRAMEWORK_OPT_EXAMPLE_KEY",
                "AGENT_LEARNING_VOICE_STREAMING_OPT_EXAMPLE_KEY",
                "AGENT_LEARNING_REDTEAM_OPT_EXAMPLE_KEY",
                "AGENT_LEARNING_WORKSPACE_OBSERVABILITY_OPT_EXAMPLE_KEY",
                "AGENT_LEARNING_AGENT_INTEGRATION_OPT_EXAMPLE_KEY",
                "AGENT_LEARNING_MULTI_AGENT_FRAMEWORK_HANDOFF_OPT_EXAMPLE_KEY",
                "AGENT_LEARNING_OPTIMIZER_GOVERNANCE_OPT_EXAMPLE_KEY",
                "AGENT_LEARNING_AGENT_CONTROL_PLANE_OPT_EXAMPLE_KEY",
                "AGENT_LEARNING_BROWSER_CUA_OPT_EXAMPLE_KEY",
                "AGENT_LEARNING_FRAMEWORK_CERT_OPT_EXAMPLE_KEY",
                "AGENT_LEARNING_AUTONOMOUS_REDTEAM_OPT_EXAMPLE_KEY",
                "AGENT_LEARNING_MULTIMODAL_IMAGE_OPT_EXAMPLE_KEY",
                "AGENT_LEARNING_SDK_PERSISTENT_REDTEAM_KEY",
                "AGENT_LEARNING_SUITE_OPT_EXAMPLE_KEY",
            ],
        ),
        (
            "run",
            "voice_streaming_realtime_manifest.json",
            "agent-learning.run.v1",
            "AGENT_LEARNING_VOICE_STREAMING_EXAMPLE_KEY",
        ),
    ],
)
def test_shipped_examples_execute_through_unified_cli(
    command: str,
    example: str,
    kind: str,
    required_env: str | list[str] | None,
    tmp_path,
    monkeypatch,
):
    for env_key in [required_env] if isinstance(required_env, str) else required_env or []:
        monkeypatch.setenv(env_key, f"real-local-{env_key.lower()}")

    output_path = tmp_path / f"{command}.json"
    junit_path = tmp_path / f"{command}.junit.xml"
    sarif_path = tmp_path / f"{command}.sarif.json"
    markdown_path = tmp_path / f"{command}.md"

    args = [
        command,
        str(EXAMPLES / example),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ]
    if command == "run":
        args.append("--no-eval")
    if command == "eval-artifact":
        args.extend(["--config", str(EXAMPLES / "artifact_task_eval_config.json")])
    if command == "eval-task":
        args.extend(["--config", str(EXAMPLES / "task_evidence_eval_config.json")])

    exit_code = main(args)

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == kind
    assert payload["status"] == "passed"
    assert payload["exit_code"] == 0
    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    assert json.loads(sarif_path.read_text(encoding="utf-8"))["version"] == "2.1.0"
    assert markdown_path.read_text(encoding="utf-8").strip()

    if command == "optimize":
        assert payload["summary"]["optimization_score"] >= 0.9
        assert payload["optimization"]["best_config"]
    if command == "optimize-eval":
        assert payload["summary"]["optimization_score"] == pytest.approx(1.0)
        assert payload["optimization"]["best_config"]
    if command == "optimize-suite":
        assert payload["summary"]["optimization_score"] == pytest.approx(1.0)
        assert payload["optimization"]["best_config"]["jobs"][0]["command"] == "suite"
        assert payload["optimization"]["suite_optimization"]["source"] == (
            "agent_learning_suite"
        )
    if command == "eval" and example == "artifact_task_eval_suite.json":
        assert payload["summary"]["assertion_count"] == 8
        case = payload["eval_suite"]["cases"][0]
        assert case["provider_type"] == "artifact"
        assert {item["type"] for item in case["assertions"]} == {
            "json_path_equals",
            "json_path_gte",
        }
        assert '"task_completion": 1.0' in case["output"]
        assert '"verification_status": "approved"' in case["output"]
        assert '"canary_exfiltrated": false' in case["output"]
    if command == "eval-artifact":
        assert payload["summary"]["report_source"] == "report"
        assert payload["summary"]["source_kind"] == "agent-learning.run.v1"
        assert payload["summary"]["score"] >= 0.9
        assert payload["summary"]["metric_averages"]["task_completion"] >= 0.9
        assert payload["source"]["path"].endswith("refund_task_run.json")
    if command == "eval-task":
        assert payload["summary"]["report_source"] == "report"
        assert payload["summary"]["source_kind"] == "agent-learning.task-evidence.v1"
        assert payload["summary"]["score"] >= 0.9
        assert payload["summary"]["metric_averages"]["task_completion"] >= 0.9
        assert payload["summary"]["metric_averages"]["world_contract_quality"] >= 0.9
        assert payload["source"]["path"].endswith("task_evidence.json")
    if command == "suite":
        assert payload["summary"]["job_count"] == 24
        assert payload["summary"]["passed_count"] == 24
        assert payload["summary"]["score"] == pytest.approx(1.0)
        assert payload["summary"]["capability_gate_passed"] is True
        assert payload["summary"]["missing_required_capabilities"] == {}
        capabilities = payload["summary"]["capabilities"]
        required_capabilities = payload["summary"]["required_capabilities"]
        assert set(capabilities["commands"]) == {
            "action_run",
            "eval",
            "eval_artifact",
            "optimize",
            "optimize_eval",
            "optimize_suite",
            "redteam",
            "run",
            "suite",
        }
        assert set(capabilities["result_kinds"]) == {
            "agent_learning.action_run.v1",
            "agent_learning.eval.v1",
            "agent_learning.artifact_evaluation.v1",
            "agent_learning.eval_optimization.v1",
            "agent_learning.optimization.v1",
            "agent_learning.redteam.v1",
            "agent_learning.run.v1",
            "agent_learning.suite.v1",
            "agent_learning.suite_optimization.v1",
        }
        assert {
            "adversarial_attack_pack",
            "agent_control_plane",
            "agent_integration",
            "autonomy_loop",
            "browser_cua",
            "framework_capability",
            "framework_trace",
            "multi_agent_room",
            "multimodal_image",
            "optimizer_trace",
            "persistent_state_attack",
            "red_team_campaign",
            "streaming_trace",
            "voice",
            "world_orchestration_replay",
        } <= set(capabilities["environment_types"])
        assert {
            "agent_integration_manifest",
            "browser",
            "framework_capability_matrix",
            "framework_runtime",
            "optimizer_society_trace",
            "red_team_campaign",
            "streaming_trace",
            "voice",
            "world_contract",
        } <= set(capabilities["environment_state_keys"])
        assert {"artifact", "bland", "livekit", "retell", "twilio", "vapi"} <= set(
            capabilities["providers"]
        )
        assert {
            "autogen",
            "crewai",
            "custom_refund_orchestrator",
            "langchain",
            "langgraph",
            "llamaindex",
            "livekit",
            "openai_agents",
            "pipecat",
            "pydantic_ai",
        } <= set(
            capabilities["frameworks"]
        )
        assert {"chat", "phone", "sip", "voice", "webrtc", "websocket"} <= set(
            capabilities["channels"]
        )
        assert {
            "agent_integration_quality",
            "browser_action_outcome",
            "eval_assertions",
            "framework_capability_quality",
            "framework_runtime_contract",
            "framework_transcript_quality",
            "multi_agent_coordination_quality",
            "multimodal_faithfulness",
            "optimizer_trace_quality",
            "persistent_state_attack_coverage",
            "persistent_state_attack_quality",
            "red_team_campaign_quality",
            "voice_trace_coverage",
            "world_contract_quality",
        } <= set(capabilities["metrics"])
        for capability, values in required_capabilities.items():
            assert set(values) <= set(capabilities[capability])
        assert [child["command"] for child in payload["children"]] == [
            "run",
            "suite",
            "optimize_suite",
            "optimize",
            "optimize",
            "eval",
            "eval",
            "eval_artifact",
            "action_run",
            "redteam",
            "run",
            "optimize_eval",
            "optimize",
            "optimize",
            "optimize",
            "optimize",
            "optimize",
            "optimize",
            "optimize",
            "optimize",
            "optimize",
            "optimize",
            "optimize",
            "optimize",
        ]
        assert {child["kind"] for child in payload["children"]} == {
            "agent-learning.run.v1",
            "agent-learning.suite.v1",
            "agent-learning.eval.v1",
            "agent-learning.artifact-evaluation.v1",
            "agent-learning.action-run.v1",
            "agent-learning.redteam.v1",
            "agent-learning.eval-optimization.v1",
            "agent-learning.optimization.v1",
            "agent-learning.suite-optimization.v1",
        }
        action_child = next(
            child
            for child in payload["children"]
            if child["id"] == "artifact-action-report"
        )
        assert action_child["kind"] == "agent-learning.action-run.v1"
        assert action_child["status"] == "passed"
        assert action_child["result"]["summary"]["action_id"] == (
            "report_orchestration_strategy"
        )
        assert action_child["result"]["summary"]["output_completion_rate"] == (
            pytest.approx(1.0)
        )
        assert set(action_child["result"]["logs"]) == {
            "stdout",
            "stderr",
            "stdout_bytes",
            "stderr_bytes",
        }
        assert any(
            path.endswith("artifacts/action-loop/action-run.json")
            for path in action_child["outputs_written"]
        )
        nested = next(
            child
            for child in payload["children"]
            if child["id"] == "multi-framework-adapter-suite"
        )
        assert nested["kind"] == "agent-learning.suite.v1"
        assert nested["result"]["summary"]["commands"] == {"run": 10}
        assert [child["id"] for child in nested["result"]["children"]] == [
            "langchain-runnable",
            "langgraph-state-graph",
            "llamaindex-chat-engine",
            "openai-agents-runner",
            "autogen-agent-chat",
            "crewai-crew",
            "pydantic-ai-agent",
            "pipecat-voice-pipeline",
            "livekit-realtime-agent",
            "custom-refund-orchestrator",
        ]
        custom_framework_optimizer = next(
            child
            for child in payload["children"]
            if child["id"] == "custom-framework-adapter-optimizer"
        )
        assert custom_framework_optimizer["kind"] == "agent-learning.optimization.v1"
        assert (
            custom_framework_optimizer["result"]["optimization"]["best_config"]["agent"][
                "method"
            ]
            == "execute_task"
        )
        assert (
            custom_framework_optimizer["result"]["optimization"]["best_config"]["agent"][
                "input_mode"
            ]
            == "dict"
        )
        social_memory_optimizer = next(
            child
            for child in payload["children"]
            if child["id"] == "social-memory-framework-optimizer"
        )
        assert social_memory_optimizer["kind"] == "agent-learning.optimization.v1"
        assert social_memory_optimizer["result"]["optimization"]["optimizer_trace"][
            "optimizer"
        ] == "AgentSocialMemoryOptimizer"
        assert (
            social_memory_optimizer["result"]["optimization"]["best_config"]["agent"][
                "method"
            ]
            == "execute_task"
        )
    if command in {"run", "eval", "redteam"}:
        assert payload["summary"]["case_count"] >= 1


def test_task_evidence_suite_runs_eval_task_child(tmp_path):
    output_path = tmp_path / "task-evidence-suite.json"
    junit_path = tmp_path / "task-evidence-suite.junit.xml"
    sarif_path = tmp_path / "task-evidence-suite.sarif.json"
    markdown_path = tmp_path / "task-evidence-suite.md"

    exit_code = main([
        "suite",
        str(EXAMPLES / "task_evidence_suite.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.suite.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["job_count"] == 1
    assert payload["summary"]["passed_count"] == 1
    assert payload["summary"]["capability_gate_passed"] is True
    assert payload["summary"]["missing_required_capabilities"] == {}
    assert payload["summary"]["commands"] == {"eval_task": 1}
    child = payload["children"][0]
    assert child["command"] == "eval_task"
    assert child["kind"] == "agent-learning.artifact-evaluation.v1"
    assert child["result"]["summary"]["source_kind"] == (
        "agent-learning.task-evidence.v1"
    )
    assert child["result"]["summary"]["score"] >= 0.9
    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    assert json.loads(sarif_path.read_text(encoding="utf-8"))["runs"][0]["results"] == []
    assert "agent-learning-task-evidence-suite" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_regression_artifact_suite_example_runs_artifact_lifecycle(tmp_path):
    output_path = tmp_path / "regression-artifact-suite.json"
    junit_path = tmp_path / "regression-artifact-suite.junit.xml"
    sarif_path = tmp_path / "regression-artifact-suite.sarif.json"
    markdown_path = tmp_path / "regression-artifact-suite.md"

    exit_code = main([
        "suite",
        str(EXAMPLES / "regression_artifact_suite.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.suite.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["job_count"] == 5
    assert payload["summary"]["passed_count"] == 5
    assert payload["summary"]["capability_gate_passed"] is True
    assert payload["summary"]["missing_required_capabilities"] == {}
    assert [child["command"] for child in payload["children"]] == [
        "baseline",
        "compare",
        "report",
        "promote_to_regression",
        "replay",
    ]
    assert {child["kind"] for child in payload["children"]} == {
        "agent-learning.baseline.v1",
        "agent-learning.compare.v1",
        "agent-learning.report.v1",
        "agent-learning.regression-promotion.v1",
        "agent-learning.replay.v1",
    }
    assert payload["children"][3]["result"]["summary"]["promoted_finding_count"] == 1
    assert payload["children"][4]["result"]["summary"]["replay_pass_rate"] == 1.0
    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    assert json.loads(sarif_path.read_text(encoding="utf-8"))["runs"][0]["results"] == []
    assert "agent-learning-regression-artifact-suite" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_eval_cli_bridge_exposes_vendored_evaluation_management_cli(
    tmp_path,
    capsys,
):
    exit_code = main(["eval-cli", "list", "categories", "--format", "json"])

    assert exit_code == 0
    categories = json.loads(capsys.readouterr().out)
    assert {"name": "safety", "count": 7} in categories
    assert {"name": "rag", "count": 6} in categories

    project_dir = tmp_path / "eval-project"
    exit_code = main([
        "eval-cli",
        "init",
        str(project_dir),
        "--template",
        "basic",
        "--force",
    ])

    assert exit_code == 0
    assert (project_dir / "fi-evaluation.yaml").exists()
    assert (project_dir / "data" / "test_cases.json").exists()
    assert (project_dir / "results" / ".gitignore").exists()


def test_agent_learn_init_optimize_scaffold_uses_unified_cli(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AGENT_LEARNING_INIT_TEST_KEY", "real-local-init-key")
    project_dir = tmp_path / "agent-learning-project"
    init_output = tmp_path / "init.json"
    artifacts_dir = project_dir / "artifacts"
    optimize_output = artifacts_dir / "optimization.json"
    optimize_junit = artifacts_dir / "optimization.junit.xml"
    optimize_sarif = artifacts_dir / "optimization.sarif.json"
    optimize_markdown = artifacts_dir / "optimization.md"
    optimization_report = artifacts_dir / "optimization-report.json"
    optimization_report_markdown = artifacts_dir / "optimization-report.md"
    promotion_output = artifacts_dir / "promotion.json"
    promotion_report = artifacts_dir / "promotion-report.json"
    promotion_report_markdown = artifacts_dir / "promotion-report.md"
    regression_manifest = project_dir / "regressions" / "optimized-regression.json"
    replay_output = artifacts_dir / "replay.json"
    replay_junit = artifacts_dir / "replay.junit.xml"
    replay_sarif = artifacts_dir / "replay.sarif.json"
    replay_markdown = artifacts_dir / "replay.md"
    replay_report = artifacts_dir / "replay-report.json"
    replay_report_markdown = artifacts_dir / "replay-report.md"
    manifest_path = project_dir / "manifests" / "optimize.json"

    exit_code = main([
        "init",
        str(project_dir),
        "--preset",
        "optimize",
        "--name",
        "refund-agent",
        "--required-env",
        "AGENT_LEARNING_INIT_TEST_KEY",
        "--force",
        "--output",
        str(init_output),
    ])

    assert exit_code == 0
    payload = json.loads(init_output.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.init.v1"
    assert payload["schema_version"] == "agent-learning.cli.v1"
    assert payload["summary"]["preset"] == "optimize"
    assert payload["summary"]["required_env"] == ["AGENT_LEARNING_INIT_TEST_KEY"]
    assert payload["init"]["next_commands"] == [
        f"agent-learn optimize {manifest_path} --dry-run",
        (
            f"agent-learn optimize {manifest_path} --output {optimize_output} "
            f"--junit {optimize_junit} --sarif {optimize_sarif} "
            f"--markdown {optimize_markdown}"
        ),
        (
            f"agent-learn report {optimize_output} "
            f"--output {optimization_report} "
            f"--markdown {optimization_report_markdown}"
        ),
        (
            f"agent-learn promote-to-regression {optimize_output} "
            f"--output {promotion_output} --manifest {regression_manifest} "
            "--min-level note --max-findings 1 "
            "--required-env AGENT_LEARNING_INIT_TEST_KEY"
        ),
        (
            f"agent-learn report {promotion_output} "
            f"--output {promotion_report} --markdown {promotion_report_markdown}"
        ),
        (
            f"agent-learn replay {regression_manifest} "
            f"--output {replay_output} --junit {replay_junit} "
            f"--sarif {replay_sarif} --markdown {replay_markdown}"
        ),
        (
            f"agent-learn report {replay_output} "
            f"--output {replay_report} --markdown {replay_report_markdown}"
        ),
    ]

    readme = (project_dir / "README.md").read_text(encoding="utf-8")
    assert "Generated by `agent-learn init`." in readme
    assert "agent-learn replay manifests" in readme
    assert "## Optimization Lifecycle" in readme
    assert "agent-learn promote-to-regression" in readme
    assert "regressions/optimized-regression.json" in readme
    assert "agent-simulate" not in readme
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == "agent-learning.optimization.v1"
    assert set(manifest["optimization"]["target"]["search_space"]) == {
        "agent.responses.0.tool_calls",
        "simulation.environments.0.data.transitions",
    }

    exit_code = main([
        "optimize",
        str(manifest_path),
        "--output",
        str(optimize_output),
        "--junit",
        str(optimize_junit),
        "--sarif",
        str(optimize_sarif),
        "--markdown",
        str(optimize_markdown),
    ])

    assert exit_code == 0
    optimized = json.loads(optimize_output.read_text(encoding="utf-8"))
    assert optimized["kind"] == "agent-learning.optimization.v1"
    assert optimized["status"] == "passed"
    assert optimized["summary"]["optimization_score"] >= 0.95
    best_config = optimized["optimization"]["best_config"]
    assert best_config["agent"]["responses"][0]["tool_calls"][0]["name"] == (
        "apply_world_transition"
    )
    assert best_config["simulation"]["environments"][0]["data"]["transitions"][0][
        "id"
    ] == "approve_refund"
    best_history = max(
        optimized["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["metrics"]["world_contract_quality"] == pytest.approx(1.0)
    assert "failures=\"0\"" in optimize_junit.read_text(encoding="utf-8")
    assert json.loads(optimize_sarif.read_text(encoding="utf-8"))["runs"][0][
        "results"
    ] == []
    assert "refund-agent-task-world-optimize" in optimize_markdown.read_text(
        encoding="utf-8"
    )

    exit_code = main([
        "report",
        str(optimize_output),
        "--output",
        str(optimization_report),
        "--markdown",
        str(optimization_report_markdown),
    ])
    assert exit_code == 0

    optimization_card = json.loads(
        optimization_report.read_text(encoding="utf-8")
    )["report"]["optimizer_replay"]
    optimization_diagnosis = json.loads(
        optimization_report.read_text(encoding="utf-8")
    )["report"]["harness_diagnosis"]
    assert optimization_card["kind"] == "optimization_result"
    assert optimization_card["source_manifest_path"] == str(manifest_path)
    assert {action["id"] for action in optimization_card["actions"]} >= {
        "rerun_optimization",
        "promote_to_regression",
        "report_artifact",
    }
    assert {"tooling", "verification"} <= {
        layer["layer"]
        for layer in optimization_diagnosis["layers"]
    }
    optimization_action_ids = {
        action["id"]
        for action in optimization_diagnosis["actions"]
    }
    assert {
        "report_harness_diagnosis",
        "rerun_optimization_for_diagnosed_layers",
        "promote_diagnosed_regression",
    } <= optimization_action_ids
    assert {"verification"} <= set(
        next(
            action
            for action in optimization_diagnosis["actions"]
            if action["id"] == "rerun_optimization_for_diagnosed_layers"
        )["target_layers"]
    )

    exit_code = main([
        "promote-to-regression",
        str(optimize_output),
        "--output",
        str(promotion_output),
        "--manifest",
        str(regression_manifest),
        "--min-level",
        "note",
        "--max-findings",
        "1",
        "--required-env",
        "AGENT_LEARNING_INIT_TEST_KEY",
    ])
    assert exit_code == 0

    promotion = json.loads(promotion_output.read_text(encoding="utf-8"))
    assert promotion["status"] == "passed"
    assert promotion["summary"]["promotion_kind"] == "optimized_manifest"
    assert promotion["summary"]["promoted_manifest_count"] == 1
    assert promotion["summary"]["best_candidate_id"] == optimized["summary"][
        "best_candidate_id"
    ]
    promoted = json.loads(regression_manifest.read_text(encoding="utf-8"))
    assert promoted["version"] == "agent-learning.run.v1"
    assert promoted["required_env"] == ["AGENT_LEARNING_INIT_TEST_KEY"]
    promoted_env_types = {
        environment["type"]
        for environment in promoted["simulation"]["environments"]
    }
    assert {"world_contract", "optimizer_trace"} <= promoted_env_types
    assert promoted["metadata"]["regression"]["promotion_kind"] == (
        "optimized_manifest"
    )

    exit_code = main([
        "report",
        str(promotion_output),
        "--output",
        str(promotion_report),
        "--markdown",
        str(promotion_report_markdown),
    ])
    assert exit_code == 0

    promotion_card = json.loads(
        promotion_report.read_text(encoding="utf-8")
    )["report"]["optimizer_replay"]
    assert promotion_card["kind"] == "promotion_manifest"
    assert promotion_card["promotion_kind"] == "optimized_manifest"
    assert promotion_card["artifacts"]["promoted_manifest"]["name"] == promoted["name"]
    assert {action["id"] for action in promotion_card["actions"]} >= {
        "recreate_promotion",
        "replay_promoted_manifest",
        "export_promoted_manifest",
    }
    assert "### Promoted Manifest" in promotion_report_markdown.read_text(
        encoding="utf-8"
    )

    exit_code = main([
        "replay",
        str(regression_manifest),
        "--output",
        str(replay_output),
        "--junit",
        str(replay_junit),
        "--sarif",
        str(replay_sarif),
        "--markdown",
        str(replay_markdown),
    ])
    assert exit_code == 0

    replay = json.loads(replay_output.read_text(encoding="utf-8"))
    assert replay["status"] == "passed"
    assert replay["summary"]["replay_pass_rate"] == pytest.approx(1.0)
    replay_child = replay["replay"]["manifests"][0]
    assert replay_child["status"] == "passed"
    assert replay_child["summary"]["metric_averages"][
        "world_contract_quality"
    ] == pytest.approx(1.0)
    assert "failures=\"0\"" in replay_junit.read_text(encoding="utf-8")
    assert not [
        result
        for result in json.loads(replay_sarif.read_text(encoding="utf-8"))["runs"][0][
            "results"
        ]
        if result.get("level") == "error"
    ]
    assert "### Replay Metrics" in replay_markdown.read_text(encoding="utf-8")

    exit_code = main([
        "report",
        str(replay_output),
        "--output",
        str(replay_report),
        "--markdown",
        str(replay_report_markdown),
    ])
    assert exit_code == 0

    replay_card = json.loads(replay_report.read_text(encoding="utf-8"))["report"][
        "replay"
    ]
    assert replay_card["kind"] == "replay_metrics"
    assert replay_card["manifest_count"] == 1
    assert replay_card["replay_pass_rate"] == pytest.approx(1.0)
    assert {action["id"] for action in replay_card["actions"]} == {
        "rerun_replay",
        "report_artifact",
    }


def test_agent_learn_simulate_init_uses_unified_engine_defaults(tmp_path):
    project_dir = tmp_path / "delegated-agent-learning-project"
    output_path = tmp_path / "delegated-init.json"

    exit_code = main([
        "simulate",
        "init",
        str(project_dir),
        "--preset",
        "redteam",
        "--required-env",
        "AGENT_LEARNING_DELEGATED_INIT_KEY",
        "--force",
        "--output",
        str(output_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.init.v1"
    assert payload["schema_version"] == "agent-learning.cli.v1"
    assert payload["init"]["next_commands"] == [
        (
            f"agent-learn redteam {project_dir / 'manifests' / 'redteam.json'} "
            f"--output {project_dir / 'artifacts' / 'redteam.json'}"
        )
    ]
    readme = (project_dir / "README.md").read_text(encoding="utf-8")
    assert "Generated by `agent-learn init`." in readme
    assert "agent-learn replay manifests" in readme
    assert "agent-simulate" not in readme
    manifest = json.loads(
        (project_dir / "manifests" / "redteam.json").read_text(encoding="utf-8")
    )
    campaign = next(
        environment
        for environment in manifest["simulation"]["environments"]
        if environment["type"] == "red_team_campaign"
    )
    run = campaign["data"]["runs"][0]
    assert run["id"] == "agent-learning-local"
    assert run["framework"] == "fi.alk"


def test_agent_learn_init_all_scaffold_runs_trinity_suite(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AGENT_LEARNING_INIT_ALL_KEY", "real-local-init-all-key")
    project_dir = tmp_path / "agent-learning-all-project"
    init_output = tmp_path / "init-all.json"
    suite_output = project_dir / "artifacts" / "suite.json"
    suite_junit = project_dir / "artifacts" / "suite.junit.xml"
    suite_sarif = project_dir / "artifacts" / "suite.sarif.json"
    suite_markdown = project_dir / "artifacts" / "suite.md"
    trust_output = project_dir / "artifacts" / "suite-trust.json"

    exit_code = main([
        "init",
        str(project_dir),
        "--preset",
        "all",
        "--name",
        "refund-agent",
        "--required-env",
        "AGENT_LEARNING_INIT_ALL_KEY",
        "--force",
        "--output",
        str(init_output),
    ])

    assert exit_code == 0
    payload = json.loads(init_output.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.init.v1"
    assert payload["summary"]["files_written_count"] == 13
    assert payload["init"]["next_commands"] == [
        (
            f"agent-learn suite {project_dir / 'manifests' / 'suite.json'} "
            f"--output {project_dir / 'artifacts' / 'suite.json'} "
            f"--junit {project_dir / 'artifacts' / 'suite.junit.xml'} "
            f"--sarif {project_dir / 'artifacts' / 'suite.sarif.json'} "
            f"--markdown {project_dir / 'artifacts' / 'suite.md'}"
        )
    ]
    assert {
        "run.json",
        "redteam.json",
        "optimize.json",
        "eval.json",
        "artifact_task_eval_suite.json",
        "artifact_task_eval_config.json",
        "eval_suite_optimization.json",
        "world_model_optimization.json",
        "suite.json",
    } <= {
        path.name
        for path in (project_dir / "manifests").iterdir()
    }
    world_model_manifest = json.loads(
        (project_dir / "manifests" / "world_model_optimization.json").read_text(
            encoding="utf-8",
        )
    )
    assert world_model_manifest["optimization"]["target"]["metadata"][
        "task_kind"
    ] == "world_model"
    assert world_model_manifest["optimization"]["target"]["metadata"][
        "world_model"
    ]["requires_external_service"] is False
    scaffold_suite_manifest = json.loads(
        (project_dir / "manifests" / "suite.json").read_text(encoding="utf-8")
    )
    assert scaffold_suite_manifest["optimizer_governance_policy"] == {
        "require_optimizer_governance": True,
        "min_governed": 1,
    }
    artifact_suite = json.loads(
        (project_dir / "manifests" / "artifact_task_eval_suite.json").read_text(
            encoding="utf-8",
        )
    )
    assert {item["type"] for item in artifact_suite["tests"][0]["assert"]} == {
        "json_path_equals",
        "json_path_gte",
    }

    exit_code = main([
        "suite",
        str(project_dir / "manifests" / "suite.json"),
        "--output",
        str(suite_output),
        "--junit",
        str(suite_junit),
        "--sarif",
        str(suite_sarif),
        "--markdown",
        str(suite_markdown),
    ])

    assert exit_code == 0
    suite = json.loads(suite_output.read_text(encoding="utf-8"))
    assert suite["kind"] == "agent-learning.suite.v1"
    assert suite["status"] == "passed"
    assert suite["summary"]["trust_certificate_verdict"] == "approved"
    assert suite["summary"]["trust_certificate_assurance_level"] == (
        "l3_trinity_governed"
    )
    assert suite["summary"]["trust_certificate_promotion_ready"] is True
    assert suite["trust_certificate"]["kind"] == (
        "agent-learning.suite.trust-certificate.v1"
    )
    assert suite["trust_certificate"]["verdict"] == "approved"
    assert suite["trust_certificate"]["promotion_ready"] is True
    assert suite["trust_certificate"]["coverage"] == {
        "simulation": True,
        "evaluation": True,
        "redteam": True,
        "optimization": True,
    }
    assert suite["trust_certificate"]["failed_gate_ids"] == []
    assert suite["trust_certificate"]["conditional_gate_ids"] == []
    assert suite["summary"]["score"] == pytest.approx(1.0)
    assert suite["summary"]["job_count"] == 9
    assert suite["summary"]["passed_count"] == 9
    assert suite["summary"]["failed_count"] == 0
    assert suite["summary"]["capability_gate_passed"] is True
    assert suite["summary"]["evidence_gate_passed"] is True
    assert suite["summary"]["optimizer_governance_gate_passed"] is True
    assert suite["summary"]["optimizer_governance_target_count"] == 2
    assert suite["summary"]["optimizer_governance_governed_count"] == 2
    assert suite["summary"]["optimizer_governance_passed_count"] == 2
    assert suite["summary"]["optimizer_governance_failed_count"] == 0
    assert suite["summary"]["optimizer_governance_missing_count"] == 0
    assert suite["optimizer_governance"]["status"] == "passed"
    assert suite["optimizer_governance"]["governed_child_ids"] == [
        "task-world-optimizer",
        "world-model-optimizer",
    ]
    assert suite["summary"]["admitted_evidence_count"] == 6
    assert suite["summary"]["non_admitted_evidence_count"] == 3
    assert suite["summary"]["frozen_evidence_count"] == 9
    assert suite["summary"]["unfrozen_evidence_count"] == 0
    assert suite["summary"]["admitted_frozen_evidence_count"] == 6

    trust_exit_code = main([
        "trust",
        str(suite_output),
        "--output",
        str(trust_output),
    ])

    assert trust_exit_code == 0
    trust = json.loads(trust_output.read_text(encoding="utf-8"))
    assert trust["kind"] == "agent-learning.suite.trust-verification.v1"
    assert trust["status"] == "passed"
    assert trust["required_verdict"] == "approved"
    assert trust["observed_verdict"] == "approved"
    assert trust["promotion_ready"] is True
    assert trust["summary"] == {
        "certificate_present": True,
        "certificate_kind_passed": True,
        "verdict_rank_passed": True,
        "promotion_gate_passed": True,
        "finding_count": 0,
    }
    assert trust["findings"] == []
    assert suite["evidence_admission"]["by_status"] == {
        "admitted": 6,
        "fixture": 3,
    }
    assert {
        child["kind"]
        for child in suite["children"]
    } == {
        "agent-learning.run.v1",
        "agent-learning.eval.v1",
        "agent-learning.artifact-evaluation.v1",
        "agent-learning.action-run.v1",
        "agent-learning.redteam.v1",
        "agent-learning.eval-optimization.v1",
        "agent-learning.optimization.v1",
    }
    action_child = next(
        child
        for child in suite["children"]
        if child["id"] == "artifact-action-report"
    )
    assert action_child["kind"] == "agent-learning.action-run.v1"
    assert action_child["status"] == "passed"
    assert action_child["result"]["summary"]["action_id"] == (
        "report_orchestration_strategy"
    )
    assert action_child["result"]["summary"]["output_completion_rate"] == pytest.approx(
        1.0,
    )
    assert action_child["evidence"]["status"] == "fixture"
    assert action_child["evidence"]["freeze"]["content_addressed"] is True
    assert action_child["evidence"]["freeze"]["outputs"]
    assert any(
        path.endswith("artifacts/action-loop/action-run.json")
        for path in action_child["outputs_written"]
    )
    world_model_child = next(
        child
        for child in suite["children"]
        if child["id"] == "world-model-optimizer"
    )
    assert world_model_child["kind"] == "agent-learning.optimization.v1"
    assert world_model_child["status"] == "passed"
    assert world_model_child["summary"]["optimization_score"] == pytest.approx(1.0)
    best_env = world_model_child["result"]["optimization"]["best_config"][
        "simulation"
    ]["environments"][0]
    assert best_env["data"]["metadata"]["candidate_profile"] == (
        "l3_evolver_verifiable_world_model"
    )
    assert best_env["data"]["world_model"]["requires_external_service"] is False
    assert 'failures="0"' in suite_junit.read_text(encoding="utf-8")
    assert json.loads(suite_sarif.read_text(encoding="utf-8"))["version"] == "2.1.0"
    assert "refund-agent-trinity-suite" in suite_markdown.read_text(
        encoding="utf-8",
    )
    assert "## Trust Certificate" in suite_markdown.read_text(encoding="utf-8")
    assert "- Verdict: `approved`" in suite_markdown.read_text(encoding="utf-8")


def test_agent_learn_suite_can_require_optimizer_governance(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_OPTIMIZE_EXAMPLE_KEY",
        "real-local-suite-governance-gate-key",
    )
    suite_manifest = {
        "version": "agent-learning.suite.v1",
        "name": "optimizer-governance-required-suite",
        "jobs": [
            {
                "id": "dry-run-optimizer",
                "command": "optimize",
                "path": str(EXAMPLES / "optimization_manifest.json"),
            }
        ],
    }
    suite_path = tmp_path / "optimizer-governance-required-suite.json"
    output_path = tmp_path / "optimizer-governance-required-output.json"
    trust_output = tmp_path / "optimizer-governance-required-trust.json"
    suite_path.write_text(
        json.dumps(suite_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    exit_code = main([
        "suite",
        str(suite_path),
        "--dry-run",
        "--require-optimizer-governance",
        "--output",
        str(output_path),
    ])

    assert exit_code == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["summary"]["trust_certificate_verdict"] == "rejected"
    assert payload["summary"]["trust_certificate_promotion_ready"] is False
    assert payload["trust_certificate"]["verdict"] == "rejected"
    assert payload["trust_certificate"]["promotion_ready"] is False
    assert "execution" in payload["trust_certificate"]["failed_gate_ids"]
    assert "optimizer_governance" in (
        payload["trust_certificate"]["conditional_gate_ids"]
    )
    assert payload["summary"]["optimizer_governance_gate_passed"] is False
    assert payload["summary"]["optimizer_governance_target_count"] == 1
    assert payload["summary"]["optimizer_governance_governed_count"] == 0
    assert payload["summary"]["optimizer_governance_missing_count"] == 1
    assert payload["optimizer_governance"]["missing_child_ids"] == [
        "dry-run-optimizer"
    ]
    assert {
        finding["type"]
        for finding in payload["findings"]
    } >= {
        "suite_optimizer_governance_missing",
        "suite_optimizer_governance_failed",
    }

    trust_exit_code = main([
        "trust",
        str(output_path),
        "--output",
        str(trust_output),
    ])

    assert trust_exit_code == 1
    trust = json.loads(trust_output.read_text(encoding="utf-8"))
    assert trust["status"] == "failed"
    assert trust["observed_verdict"] == "rejected"
    assert trust["promotion_ready"] is False
    assert {
        finding["type"]
        for finding in trust["findings"]
    } == {
        "suite_trust_certificate_verdict_too_low",
        "suite_trust_certificate_not_promotion_ready",
    }


def test_sdk_built_eval_suite_runs_through_cli_and_suite(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "AGENT_LEARNING_SDK_EVAL_SUITE_KEY",
        "real-local-sdk-eval-suite-key",
    )
    example_path = EXAMPLES / "sdk_eval_suite.py"
    spec = importlib.util.spec_from_file_location("sdk_eval_suite", example_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sdk_output = tmp_path / "sdk-eval-result.json"
    direct = module.run(sdk_output)
    manifest_path = sdk_output.with_suffix(".manifest.json")
    suite_path = sdk_output.with_suffix(".suite.json")
    assert direct["status"] == "passed"
    assert json.loads(suite_path.read_text(encoding="utf-8"))["required_env"] == []

    cli_output = tmp_path / "sdk-eval-cli.json"
    junit_path = tmp_path / "sdk-eval-cli.junit.xml"
    sarif_path = tmp_path / "sdk-eval-cli.sarif.json"
    markdown_path = tmp_path / "sdk-eval-cli.md"
    exit_code = main([
        "eval",
        str(manifest_path),
        "--output",
        str(cli_output),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])
    assert exit_code == 0
    cli_payload = json.loads(cli_output.read_text(encoding="utf-8"))
    assert cli_payload["kind"] == "agent-learning.eval.v1"
    assert cli_payload["status"] == "passed"
    assert cli_payload["summary"]["score"] == pytest.approx(1.0)
    assert cli_payload["summary"]["assertion_count"] == 2
    assert 'failures="0"' in junit_path.read_text(encoding="utf-8")
    assert json.loads(sarif_path.read_text(encoding="utf-8"))["runs"][0][
        "results"
    ] == []
    assert "sdk-local-eval-suite" in markdown_path.read_text(encoding="utf-8")

    suite_output = tmp_path / "sdk-eval-suite-result.json"
    suite_exit = main(["suite", str(suite_path), "--output", str(suite_output)])
    assert suite_exit == 0
    suite_payload = json.loads(suite_output.read_text(encoding="utf-8"))
    assert suite_payload["kind"] == "agent-learning.suite.v1"
    assert suite_payload["status"] == "passed"
    assert suite_payload["summary"]["score"] == pytest.approx(1.0)
    assert suite_payload["children"][0]["kind"] == "agent-learning.eval.v1"


def test_sdk_framework_adapter_probe_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_probe.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_probe",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-probe.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.framework-adapter-probe.v1"
    assert result["status"] == "passed"
    assert result["summary"]["runtime_trace_count"] == 1
    assert result["summary"]["call_contract_count"] == 1
    assert result["summary"]["observed_io_contract_count"] == 1
    assert result["summary"]["signature_bound_count"] == 1
    assert result["summary"]["input_keys"] == ["payload"]
    assert result["summary"]["tool_call_count"] == 1
    assert result["contract"]["framework"] == "custom_refund_orchestrator"
    assert result["contract"]["callable_signature"]["keyword_only_parameters"] == [
        "payload"
    ]
    assert result["cases"][0]["runtime_trace"]["metadata"][
        "framework_adapter_contract"
    ] == result["contract"]
    assert result["cases"][0]["runtime_trace"]["invocations"][0]["call_contract"][
        "signature_bound"
    ] is True


def test_sdk_framework_adapter_discovery_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_discovery.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_discovery",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-discovery.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.framework-adapter-discovery.v1"
    assert result["status"] == "passed"
    assert result["summary"]["top_method"] == "execute_task"
    assert result["summary"]["top_input_mode"] == "dict"
    assert result["adapter_candidates"][0]["method"] == "execute_task"
    assert result["adapter_candidates"][0]["input_mode"] == "dict"


def test_sdk_framework_adapter_probe_optimization_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_probe_optimization.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_probe_optimization",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-probe-optimization.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.optimization.v1"
    assert result["status"] == "passed"
    assert result["summary"]["framework_adapter_probe_proof_passed"] is True
    assert result["optimization_governance"]["status"] == "passed"
    best_adapter = result["optimization"]["best_config"]["adapter"]
    assert best_adapter["method"] == "execute_task"
    assert best_adapter["input_mode"] == "dict"
    assert result["framework_adapter_probe_proof"]["failed_check_ids"] == []

    report_path = tmp_path / "sdk-framework-adapter-probe-optimization-report.json"
    report_markdown_path = (
        tmp_path / "sdk-framework-adapter-probe-optimization-report.md"
    )
    assert (
        main(
            [
                "report",
                str(output_path),
                "--output",
                str(report_path),
                "--markdown",
                str(report_markdown_path),
            ]
        )
        == 0
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_markdown = report_markdown_path.read_text(encoding="utf-8")
    assert "framework_adapter_probe" in report["summary"]["sections"]
    adapter_card = report["report"]["framework_adapter_probe"]
    assert adapter_card["kind"] == "framework_adapter_probe_evidence"
    assert adapter_card["status"] == "verified"
    assert adapter_card["framework"] == "custom_refund_orchestrator"
    assert adapter_card["method"] == "execute_task"
    assert adapter_card["input_mode"] == "dict"
    assert adapter_card["local_only"] is True
    assert adapter_card["requires_external_service"] is False
    assert adapter_card["proof_status"] == "passed"
    assert adapter_card["runtime_trace_count"] == 1
    assert adapter_card["call_contract_count"] == 1
    assert adapter_card["observed_io_contract_count"] == 1
    assert adapter_card["signature_bound_count"] == 1
    assert adapter_card["callable_signature_inspectable"] is True
    assert adapter_card["tool_call_count"] == 1
    assert adapter_card["artifacts"]["proof"] == result["framework_adapter_probe_proof"]
    assert adapter_card["artifacts"]["callable_signature"]["kind"] == (
        "agent-learning.framework-adapter-callable-signature.v1"
    )
    assert adapter_card["artifacts"]["observed_io_contract"]["summary"][
        "signature_bound_count"
    ] == 1
    assert "## Framework Adapter Probe" in report_markdown
    assert "Observed I/O contracts" in report_markdown

    catalog = actions.action_catalog(result, source_path=output_path)
    framework_actions = {
        action["id"]: action
        for action in catalog["actions"]
        if action.get("source_card_path") == "framework_adapter_probe"
    }
    assert {
        "report_framework_adapter_probe",
        "export_framework_adapter_probe_proof",
        "export_framework_adapter_probe_selected_probe_report",
        "export_framework_adapter_probe_contract",
        "export_framework_adapter_probe_callable_signature",
        "export_framework_adapter_probe_observed_io_contract",
        "export_framework_adapter_probe_replay_lock",
    } <= set(framework_actions)
    assert framework_actions["export_framework_adapter_probe_proof"][
        "artifact_ref"
    ] == "report.framework_adapter_probe.artifacts.proof"

    proof_export_path = tmp_path / "framework-adapter-probe-proof.json"
    export_run = actions.run_action(
        result,
        "export_framework_adapter_probe_proof",
        source_path=output_path,
        cwd=tmp_path,
        artifact_output_path=proof_export_path,
    )
    assert export_run["kind"] == "agent-learning.action-run.v1"
    assert export_run["status"] == "passed"
    assert export_run["summary"]["source_card_path"] == "framework_adapter_probe"
    assert export_run["artifact_ref"] == (
        "report.framework_adapter_probe.artifacts.proof"
    )
    exported_proof = json.loads(proof_export_path.read_text(encoding="utf-8"))
    assert exported_proof == result["framework_adapter_probe_proof"]

    signature_export_path = tmp_path / "framework-adapter-probe-signature.json"
    signature_export_run = actions.run_action(
        result,
        "export_framework_adapter_probe_callable_signature",
        source_path=output_path,
        cwd=tmp_path,
        artifact_output_path=signature_export_path,
    )
    assert signature_export_run["status"] == "passed"
    exported_signature = json.loads(
        signature_export_path.read_text(encoding="utf-8")
    )
    assert exported_signature == adapter_card["artifacts"]["callable_signature"]


def test_sdk_framework_adapter_auto_discovery_optimization_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_auto_discovery_optimization.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_auto_discovery_optimization",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-auto-discovery-optimization.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.optimization.v1"
    assert result["status"] == "passed"
    assert result["summary"]["adapter_candidate_source"] == "discovery"
    assert result["summary"]["framework_adapter_discovery_used"] is True
    assert result["summary"]["framework_adapter_probe_proof_passed"] is True
    assert result["optimization"]["best_config"]["adapter"]["method"] == (
        "execute_task"
    )
    assert result["framework_adapter_probe_proof"]["failed_check_ids"] == []


def test_sdk_framework_adapter_probe_promotion_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_probe_promotion.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_probe_promotion",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-probe-promotion.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        output_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    assert result["summary"]["metric_averages"]["framework_runtime_contract"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"][
        "framework_adapter_contract_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "framework_adapter_call_contract_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "framework_adapter_observed_io_quality"
    ] == pytest.approx(1.0)
    assert manifest["agent"]["method"] == "execute_task"
    assert manifest["agent"]["input_mode"] == "dict"
    assert manifest["agent"]["metadata"]["promoted_from_framework_adapter_probe"] is True
    assert manifest["agent"]["metadata"]["framework_adapter_probe_proof"][
        "status"
    ] == "passed"


def test_sdk_framework_adapter_auto_discovery_promotion_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_auto_discovery_promotion.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_auto_discovery_promotion",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-auto-discovery-promotion.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        output_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    assert result["summary"]["metric_averages"]["framework_runtime_contract"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"][
        "framework_adapter_contract_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "framework_adapter_call_contract_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "framework_adapter_observed_io_quality"
    ] == pytest.approx(1.0)
    assert manifest["agent"]["method"] == "execute_task"
    assert manifest["agent"]["input_mode"] == "dict"
    assert manifest["agent"]["metadata"]["adapter_candidate_source"] == "discovery"
    assert (
        manifest["agent"]["metadata"]["framework_adapter_discovery"]["status"]
        == "passed"
    )
    assert manifest["metadata"]["framework_adapter_discovery_used"] is True


def test_sdk_framework_adapter_one_call_promotion_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_one_call_promotion.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_one_call_promotion",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-one-call-promotion.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        output_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    assert result["summary"]["metric_averages"]["framework_runtime_contract"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"][
        "framework_adapter_contract_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "framework_adapter_call_contract_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "framework_adapter_observed_io_quality"
    ] == pytest.approx(1.0)
    assert manifest["agent"]["method"] == "execute_task"
    assert manifest["agent"]["input_mode"] == "dict"
    assert manifest["agent"]["metadata"]["adapter_candidate_source"] == "discovery"
    assert manifest["evaluation"]["enabled"] is True


def test_sdk_framework_adapter_langgraph_ainvoke_promotion_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_langgraph_ainvoke_promotion.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_langgraph_ainvoke_promotion",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-langgraph-ainvoke-promotion.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        output_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    assert result["summary"]["metric_averages"]["framework_runtime_contract"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"][
        "framework_adapter_contract_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "framework_adapter_call_contract_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "framework_adapter_observed_io_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"]["framework_trace_coverage"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"]["tool_selection_accuracy"] == (
        pytest.approx(1.0)
    )
    assert manifest["agent"]["framework"] == "langgraph"
    assert manifest["agent"]["method"] == "ainvoke"
    assert manifest["agent"]["input_mode"] == "dict"
    assert manifest["agent"]["trace_runtime"] is True
    assert manifest["agent"]["metadata"]["adapter_candidate_source"] == "discovery"
    assert manifest["agent"]["metadata"]["framework_adapter_discovery_used"] is True
    assert (
        manifest["agent"]["metadata"]["framework_adapter_discovery"]["status"]
        == "passed"
    )
    assert manifest["agent"]["metadata"]["framework_adapter_probe_proof"][
        "status"
    ] == "passed"
    assert manifest["agent"]["metadata"]["framework_adapter_probe_proof"][
        "failed_check_ids"
    ] == []
    assert manifest["evaluation"]["enabled"] is True


def test_sdk_framework_adapter_langchain_invoke_promotion_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_langchain_invoke_promotion.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_langchain_invoke_promotion",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-langchain-invoke-promotion.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        output_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    assert result["summary"]["metric_averages"]["framework_runtime_contract"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"][
        "framework_adapter_contract_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "framework_adapter_call_contract_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "framework_adapter_observed_io_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"]["framework_trace_coverage"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"]["tool_selection_accuracy"] == (
        pytest.approx(1.0)
    )
    assert manifest["agent"]["framework"] == "langchain"
    assert manifest["agent"]["method"] == "invoke"
    assert manifest["agent"]["input_mode"] == "dict"
    assert manifest["agent"]["trace_runtime"] is True
    assert manifest["agent"]["metadata"]["adapter_candidate_source"] == "discovery"
    assert manifest["agent"]["metadata"]["framework_adapter_discovery_used"] is True
    assert (
        manifest["agent"]["metadata"]["framework_adapter_discovery"]["status"]
        == "passed"
    )
    assert manifest["agent"]["metadata"]["framework_adapter_probe_proof"][
        "status"
    ] == "passed"
    assert manifest["agent"]["metadata"]["framework_adapter_probe_proof"][
        "failed_check_ids"
    ] == []
    assert manifest["evaluation"]["enabled"] is True


def test_sdk_framework_adapter_pipecat_process_promotion_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_pipecat_process_promotion.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_pipecat_process_promotion",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-pipecat-process-promotion.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        output_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    assert result["summary"]["metric_averages"]["framework_runtime_contract"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"][
        "framework_adapter_contract_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "framework_adapter_call_contract_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "framework_adapter_observed_io_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"]["framework_trace_coverage"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"]["tool_selection_accuracy"] == (
        pytest.approx(1.0)
    )
    assert manifest["agent"]["framework"] == "pipecat"
    assert manifest["agent"]["method"] == "process"
    assert manifest["agent"]["input_mode"] == "dict"
    assert manifest["agent"]["trace_runtime"] is True
    assert manifest["simulation"]["modality"] == "voice"
    assert manifest["agent"]["metadata"]["adapter_candidate_source"] == "discovery"
    assert manifest["agent"]["metadata"]["framework_adapter_discovery_used"] is True
    assert (
        manifest["agent"]["metadata"]["framework_adapter_discovery"]["status"]
        == "passed"
    )
    assert manifest["agent"]["metadata"]["framework_adapter_probe_proof"][
        "status"
    ] == "passed"
    assert manifest["agent"]["metadata"]["framework_adapter_probe_proof"][
        "failed_check_ids"
    ] == []
    assert manifest["evaluation"]["enabled"] is True


def test_sdk_framework_adapter_nested_method_promotion_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_nested_method_promotion.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_nested_method_promotion",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-nested-method-promotion.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        output_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    assert result["summary"]["metric_averages"]["framework_runtime_contract"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"][
        "framework_adapter_contract_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "framework_adapter_call_contract_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "framework_adapter_observed_io_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"]["framework_trace_coverage"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"]["tool_selection_accuracy"] == (
        pytest.approx(1.0)
    )
    assert manifest["agent"]["framework"] == "openai"
    assert manifest["agent"]["method"] == "chat.completions.create"
    assert manifest["agent"]["input_mode"] == "messages"
    assert manifest["agent"]["input_key"] == "messages"
    assert manifest["agent"]["trace_runtime"] is True
    assert manifest["agent"]["metadata"]["adapter_candidate_source"] == "discovery"
    assert manifest["agent"]["metadata"]["framework_adapter_discovery_used"] is True
    assert (
        manifest["agent"]["metadata"]["framework_adapter_discovery"]["status"]
        == "passed"
    )
    proof = manifest["agent"]["metadata"]["framework_adapter_probe_proof"]
    assert proof["status"] == "passed"
    assert proof["failed_check_ids"] == []
    assert proof["method"] == "chat.completions.create"
    assert proof["input_mode"] == "messages"
    assert proof["input_key"] == "messages"
    contract = manifest["agent"]["metadata"]["framework_adapter_probe_contract"]
    assert contract["input_key"] == "messages"
    signature = contract["callable_signature"]
    assert signature["inspectable"] is True
    assert signature["keyword_only_parameters"] == ["messages"]
    assert manifest["evaluation"]["enabled"] is True

    state = result["report"]["results"][0]["metadata"]["environment_state"]
    runtime = state["framework_runtime"]
    assert runtime["summary"]["methods"] == ["chat.completions.create"]
    assert runtime["summary"]["input_modes"] == ["messages"]
    assert runtime["summary"]["input_keys"] == ["messages"]
    assert runtime["summary"]["call_styles"] == ["keyword"]
    assert state["nested_client"]["method_path"] == "chat.completions.create"
    assert state["nested_client"]["input_key"] == "messages"


def test_sdk_framework_adapter_livekit_run_session_promotion_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_livekit_run_session_promotion.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_livekit_run_session_promotion",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-livekit-run-session-promotion.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        output_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    assert result["summary"]["metric_averages"]["framework_runtime_contract"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"][
        "framework_adapter_contract_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "framework_adapter_call_contract_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "framework_adapter_observed_io_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"]["framework_trace_coverage"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"]["tool_selection_accuracy"] == (
        pytest.approx(1.0)
    )
    assert manifest["agent"]["framework"] == "livekit"
    assert manifest["agent"]["method"] == "run_session"
    assert manifest["agent"]["input_mode"] == "dict"
    assert manifest["agent"]["trace_runtime"] is True
    assert manifest["simulation"]["modality"] == "voice"
    assert manifest["agent"]["metadata"]["adapter_candidate_source"] == "discovery"
    assert manifest["agent"]["metadata"]["framework_adapter_discovery_used"] is True
    assert (
        manifest["agent"]["metadata"]["framework_adapter_discovery"]["status"]
        == "passed"
    )
    proof = manifest["agent"]["metadata"]["framework_adapter_probe_proof"]
    assert proof["status"] == "passed"
    assert proof["failed_check_ids"] == []
    assert proof["method"] == "run_session"
    assert proof["input_mode"] == "dict"
    assert manifest["evaluation"]["enabled"] is True

    result_row = result["report"]["results"][0]
    state = result_row["metadata"]["environment_state"]
    runtime = state["framework_runtime"]
    assert runtime["summary"]["methods"] == ["run_session"]
    assert runtime["summary"]["input_modes"] == ["dict"]
    assert runtime["summary"]["call_styles"] == ["positional"]
    assert state["framework_trace"]["framework"] == "livekit"
    assert state["framework_trace"]["summary"]["status"] == "passed"
    assert state["livekit_session"]["session_id"] == "livekit-session-refund-42"
    assert state["livekit_session"]["room"] == "local-refund-room"
    assert state["livekit_session"]["modality"] == "voice"
    assert state["livekit_session"]["closed"] is True
    assert "approved refund" in state["livekit_session"]["final_transcript"]
    event_types = {event["type"] for event in result_row["events"]}
    assert {"framework_trace", "livekit_session_event", "livekit_transcript"} <= (
        event_types
    )


def test_sdk_framework_adapter_one_call_run_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_one_call_run.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_one_call_run",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-one-call-run.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        output_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    assert result["summary"]["framework_adapter_direct_run"] is True
    assert result["summary"]["metric_averages"]["framework_runtime_contract"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"][
        "framework_adapter_contract_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "framework_adapter_call_contract_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "framework_adapter_observed_io_quality"
    ] == pytest.approx(1.0)
    assert manifest == result["framework_adapter_run_manifest"]
    assert manifest["agent"]["method"] == "execute_task"
    assert manifest["agent"]["input_mode"] == "dict"
    assert manifest["agent"]["metadata"]["adapter_candidate_source"] == "discovery"
    assert manifest["evaluation"]["enabled"] is True


def test_sdk_framework_adapter_trinity_suite_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_trinity_suite.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_trinity_suite",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-trinity-suite.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.suite.v1"
    assert result["status"] == "passed"
    assert result["summary"]["capability_gate_passed"] is True
    assert result["summary"]["framework_coverage_passed"] is True
    assert result["summary"]["passed_count"] == 2
    workspace = result["framework_adapter_trinity_workspace"]
    assert Path(workspace["paths"]["suite"]).exists()
    children = {child["id"]: child for child in result["children"]}
    assert children["optimized-framework-run"]["status"] == "passed"
    assert children["framework-red-team"]["status"] == "passed"
    assert children["framework-red-team"]["summary"]["metric_averages"][
        "red_team_campaign_quality"
    ] == pytest.approx(1.0)


def test_sdk_framework_adapter_trinity_suite_optimization_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_trinity_suite_optimization.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_trinity_suite_optimization",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-trinity-suite-optimization.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.suite-optimization.v1"
    assert result["status"] == "passed"
    assert result["summary"]["optimization_score"] == pytest.approx(1.0)
    assert result["optimization"]["best_config"]["jobs"][0]["command"] == "suite"
    workspace = result["framework_adapter_trinity_optimization_workspace"]
    assert Path(workspace["paths"]["suite_optimization"]).exists()
    assert workspace["suite_optimization"]["optimization"]["target"]["search_space"][
        "jobs.0"
    ][1]["command"] == "suite"


def test_sdk_framework_adapter_streaming_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_streaming.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_streaming",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-streaming.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    assert result["summary"]["metric_averages"]["streaming_trace_coverage"] == (
        pytest.approx(1.0)
    )
    manifest = result["framework_adapter_streaming_manifest"]
    assert manifest["agent"]["method"] == "astream"
    assert manifest["agent"]["input_mode"] == "dict"
    assert manifest["evaluation"]["agent_report"]["config"][
        "framework_runtime_contract"
    ]["require_streaming"] is True
    state = result["report"]["results"][0]["metadata"]["environment_state"]
    assert state["framework_runtime"]["summary"]["streamed"] is True
    assert state["streaming_trace"]["summary"]["tool_delta_count"] == 1


def test_sdk_framework_adapter_typed_output_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_typed_output.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_typed_output",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-typed-output.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    manifest = result["framework_adapter_typed_output_manifest"]
    assert manifest["agent"]["method"] == "execute_task"
    assert manifest["evaluation"]["agent_report"]["config"][
        "framework_runtime_contract"
    ]["required_state_keys"] == ["typed_output"]
    state = result["report"]["results"][0]["metadata"]["environment_state"]
    assert state["typed_output"]["decision"]["verdict"] == "approved"
    assert state["framework_runtime"]["summary"]["state_key_count"] == 1


def test_sdk_framework_adapter_keyword_inputs_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_keyword_inputs.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_keyword_inputs",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-keyword-inputs.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    manifest = result["framework_adapter_keyword_inputs_manifest"]
    assert manifest["agent"]["method"] == "kickoff"
    assert manifest["agent"]["input_key"] == "inputs"
    runtime_contract = manifest["evaluation"]["agent_report"]["config"][
        "framework_runtime_contract"
    ]
    assert runtime_contract["input_key"] == "inputs"
    assert runtime_contract["call_style"] == "keyword"
    state = result["report"]["results"][0]["metadata"]["environment_state"]
    assert state["framework_runtime"]["summary"]["input_keys"] == ["inputs"]
    assert "crewai" in state["crew_inputs"]["input"].lower()


def test_sdk_framework_adapter_side_kwargs_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_side_kwargs.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_side_kwargs",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-side-kwargs.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    manifest = result["framework_adapter_side_kwargs_manifest"]
    assert manifest["agent"]["method"] == "process_frame"
    assert manifest["agent"]["input_key"] == "frame"
    assert manifest["agent"]["input_kwargs"] == {"direction": "downstream"}
    runtime_contract = manifest["evaluation"]["agent_report"]["config"][
        "framework_runtime_contract"
    ]
    assert runtime_contract["required_input_kwargs"] == ["direction"]
    state = result["report"]["results"][0]["metadata"]["environment_state"]
    assert state["framework_runtime"]["summary"]["input_kwargs_keys"] == ["direction"]
    assert state["pipecat_frame"]["direction"] == "downstream"


def test_sdk_framework_adapter_nested_method_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_nested_method.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_nested_method",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-nested-method.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    manifest = result["framework_adapter_nested_method_manifest"]
    assert manifest["agent"]["method"] == "chat.completions.create"
    assert manifest["agent"]["input_mode"] == "messages"
    assert manifest["agent"]["input_key"] == "messages"
    runtime_contract = manifest["evaluation"]["agent_report"]["config"][
        "framework_runtime_contract"
    ]
    assert runtime_contract["method"] == "chat.completions.create"
    state = result["report"]["results"][0]["metadata"]["environment_state"]
    assert state["framework_runtime"]["summary"]["methods"] == [
        "chat.completions.create"
    ]
    assert state["nested_client"]["method_path"] == "chat.completions.create"


def test_sdk_framework_adapter_provider_response_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_provider_response.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_provider_response",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-provider-response.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    assert result["summary"]["metric_averages"]["framework_runtime_contract"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"][
        "framework_adapter_contract_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "framework_adapter_call_contract_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "framework_adapter_observed_io_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"]["framework_trace_coverage"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"]["tool_selection_accuracy"] == (
        pytest.approx(1.0)
    )
    manifest = result["framework_adapter_provider_response_manifest"]
    assert manifest["agent"]["framework"] == "openai"
    assert manifest["agent"]["method"] == "chat.completions.create"
    assert manifest["agent"]["input_mode"] == "messages"
    assert manifest["agent"]["input_key"] == "messages"
    assert manifest["agent"]["input_kwargs"] == {"model": "local-provider-model"}
    assert manifest["agent"]["trace_runtime"] is True
    assert manifest["agent"]["metadata"]["adapter_candidate_source"] == "explicit"
    assert manifest["agent"]["metadata"]["framework_adapter_discovery_used"] is False
    proof = manifest["agent"]["metadata"]["framework_adapter_probe_proof"]
    assert proof["status"] == "passed"
    assert proof["failed_check_ids"] == []
    assert proof["method"] == "chat.completions.create"
    assert proof["input_mode"] == "messages"
    assert proof["input_key"] == "messages"
    assert proof["input_kwargs_keys"] == ["model"]
    runtime_contract = manifest["evaluation"]["agent_report"]["config"][
        "framework_runtime_contract"
    ]
    assert runtime_contract["method"] == "chat.completions.create"
    assert runtime_contract["input_key"] == "messages"
    assert runtime_contract["required_tools"] == ["framework_trace_status"]
    assert runtime_contract["required_input_kwargs"] == ["model"]
    state = result["report"]["results"][0]["metadata"]["environment_state"]
    runtime = state["framework_runtime"]
    assert runtime["summary"]["methods"] == ["chat.completions.create"]
    assert runtime["summary"]["input_modes"] == ["messages"]
    assert runtime["summary"]["input_keys"] == ["messages"]
    assert runtime["summary"]["input_kwargs_keys"] == ["model"]
    assert runtime["summary"]["call_styles"] == ["keyword"]
    assert state["provider_response"]["choice_count"] == 1
    assert state["provider_response"]["tool_call_count"] == 1
    assert state["provider_response"]["model"] == "local-provider-model"
    assert state["provider_response"]["tool_names"] == ["framework_trace_status"]
    assert state["provider_response"]["usage"]["total_tokens"] == 19
    assert state["provider_response"]["finish_reasons"] == ["tool_calls"]
    output = runtime["invocations"][0]["output"]
    assert output["tool_names"] == ["framework_trace_status"]
    assert output["event_types"] == ["provider_choice", "provider_tool_call"]


def test_sdk_framework_adapter_message_history_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_message_history.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_message_history",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-message-history.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    manifest = result["framework_adapter_message_history_manifest"]
    assert manifest["agent"]["method"] == "run"
    assert manifest["agent"]["input_key"] == "task"
    config = manifest["evaluation"]["agent_report"]["config"]
    runtime_contract = config["framework_runtime_contract"]
    assert runtime_contract["required_state_keys"] == ["message_history"]
    transcript_quality = config["framework_transcript_quality"]
    assert transcript_quality["min_turns"] == 4
    assert set(transcript_quality["required_event_methods"]) >= {
        "TextMessage",
        "ToolCallRequestEvent",
        "ToolCallExecutionEvent",
        "termination",
    }
    assert transcript_quality["required_speakers"] == [
        "planner",
        "tool",
        "reviewer",
    ]
    assert transcript_quality["expected_speaker_sequence"] == [
        "planner",
        "planner",
        "tool",
        "reviewer",
    ]
    assert transcript_quality["expected_tool_sequence"] == [
        "framework_trace_status"
    ]
    assert transcript_quality["require_termination"] is True
    assert transcript_quality["termination_contains"] == ["completed"]
    assert transcript_quality["expected_state"] == {
        "message_history": {"message_count": 4}
    }
    assert config["metric_weights"]["framework_transcript_quality"] == pytest.approx(4.0)
    assert result["summary"]["metric_averages"]["framework_transcript_quality"] == (
        pytest.approx(1.0)
    )
    state = result["report"]["results"][0]["metadata"]["environment_state"]
    history = state["message_history"]
    assert history["tool_names"] == ["framework_trace_status"]
    assert history["tool_response_count"] == 1
    output = state["framework_runtime"]["invocations"][0]["output"]
    assert output["tool_names"] == ["framework_trace_status"]
    assert "ToolCallRequestEvent" in output["event_types"]


def test_sdk_framework_adapter_handoff_transcript_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_handoff_transcript.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_handoff_transcript",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-handoff-transcript.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    manifest = result["framework_adapter_handoff_transcript_manifest"]
    assert manifest["agent"]["method"] == "execute_task"
    config = manifest["evaluation"]["agent_report"]["config"]
    runtime_contract = config["framework_runtime_contract"]
    assert runtime_contract["required_state_keys"] == [
        "framework_handoffs",
        "message_history",
    ]
    transcript_quality = config["framework_transcript_quality"]
    assert transcript_quality["min_turns"] == 5
    assert set(transcript_quality["required_event_methods"]) >= {
        "handoff",
        "review",
        "reconciliation",
        "final_answer",
        "termination",
    }
    assert set(transcript_quality["required_speakers"]) >= {
        "triage_agent",
        "retrieval_agent",
        "critic_agent",
    }
    assert transcript_quality["expected_speaker_sequence"] == [
        "triage_agent",
        "retrieval_agent",
        "critic_agent",
        "critic_agent",
        "critic_agent",
    ]
    assert transcript_quality["expected_handoffs"] == [
        {
            "from": "triage_agent",
            "to": "retrieval_agent",
            "task_contains": ["Gather current refund policy evidence."],
        },
        {
            "from": "retrieval_agent",
            "to": "critic_agent",
            "task_contains": ["Review grounded refund recommendation."],
        },
    ]
    assert transcript_quality["expected_state"] == {
        "message_history": {"message_count": 5},
        "framework_handoffs": {
            "handoff_count": 2,
            "review_count": 1,
            "reconciliation_count": 1,
        },
    }
    assert transcript_quality["require_termination"] is True
    assert transcript_quality["termination_contains"] == ["completed"]
    assert config["metric_weights"]["framework_transcript_quality"] == pytest.approx(4.0)
    assert result["summary"]["metric_averages"]["framework_transcript_quality"] == (
        pytest.approx(1.0)
    )
    assert set(config["required_events"]) >= {
        "framework_handoff",
        "framework_review",
        "framework_reconciliation",
    }
    state = result["report"]["results"][0]["metadata"]["environment_state"]
    coordination = state["framework_handoffs"]
    assert coordination["handoff_count"] == 2
    assert coordination["review_count"] == 1
    assert coordination["reconciliation_count"] == 1
    event_types = set(
        state["framework_runtime"]["invocations"][0]["output"]["event_types"]
    )
    assert {
        "framework_handoff",
        "framework_review",
        "framework_reconciliation",
    } <= event_types


def test_sdk_framework_adapter_realtime_trace_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_realtime_trace.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_realtime_trace",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-realtime-trace.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    manifest = result["framework_adapter_realtime_trace_manifest"]
    assert manifest["agent"]["method"] == "run_session"
    config = manifest["evaluation"]["agent_report"]["config"]
    runtime_contract = config["framework_runtime_contract"]
    assert runtime_contract["required_state_keys"] == ["realtime_trace"]
    assert "realtime" in runtime_contract["required_signals"]
    assert set(config["required_realtime_trace"]) >= {
        "realtime_trace",
        "trace",
        "frame",
        "event",
        "tool",
        "tool_call",
        "tool_response",
        "transcript",
        "audio_frame",
        "lifecycle",
        "completion",
        "frame_type",
        "event_type",
        "data_frame",
        "control_frame",
        "inbound",
        "outbound",
        "voice",
    }
    assert set(config["realtime_trace_quality"]["required_frame_types"]) >= {
        "AudioRawFrame",
        "FunctionCallFrame",
        "FunctionCallResultFrame",
        "TranscriptionFrame",
    }
    assert set(config["realtime_trace_quality"]["required_event_types"]) >= {
        "agent_state_changed",
        "tool_execution_started",
        "tool_execution_completed",
        "transcript_final",
        "session_closed",
    }
    assert config["realtime_trace_quality"]["required_tools"] == [
        "lookup_refund_policy"
    ]
    assert config["realtime_trace_quality"]["required_directions"] == [
        "inbound",
        "outbound",
    ]
    assert config["realtime_trace_quality"]["required_modalities"] == ["voice"]
    assert config["metric_weights"]["realtime_trace_coverage"] == pytest.approx(4.0)
    assert config["metric_weights"]["realtime_trace_quality"] == pytest.approx(4.0)
    assert result["summary"]["metric_averages"]["realtime_trace_coverage"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"]["realtime_trace_quality"] == (
        pytest.approx(1.0)
    )
    assert set(config["required_events"]) >= {
        "realtime_frame",
        "realtime_tool_call",
        "realtime_tool_response",
        "realtime_transcript",
        "realtime_lifecycle",
    }
    state = result["report"]["results"][0]["metadata"]["environment_state"]
    realtime = state["realtime_trace"]
    assert realtime["frame_count"] == 5
    assert realtime["event_count"] == 5
    assert realtime["tool_call_count"] >= 1
    assert realtime["tool_response_count"] >= 1
    assert "lookup_refund_policy" in realtime["tool_names"]
    event_types = set(
        state["framework_runtime"]["invocations"][0]["output"]["event_types"]
    )
    assert {
        "realtime_frame",
        "realtime_tool_call",
        "realtime_tool_response",
        "realtime_transcript",
        "realtime_lifecycle",
    } <= event_types


def test_sdk_framework_adapter_memory_trace_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_memory_trace.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_memory_trace",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-memory-trace.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    manifest = result["framework_adapter_memory_trace_manifest"]
    assert manifest["agent"]["method"] == "ainvoke"
    config = manifest["evaluation"]["agent_report"]["config"]
    runtime_contract = config["framework_runtime_contract"]
    assert runtime_contract["required_state_keys"] == [
        "agent_memory_lineage",
        "framework_memory",
        "retrieval_memory",
    ]
    assert set(runtime_contract["required_signals"]) >= {"event", "memory", "state"}
    assert set(config["required_events"]) >= {
        "framework_memory_operation",
        "framework_memory_checkpoint",
        "framework_memory_retrieval",
        "framework_memory_record",
    }
    assert set(config["required_agent_memory_lineage"]) >= {
        "agent_memory_lineage",
        "memory_lineage",
        "memory",
        "provenance",
        "source_attribution",
        "tenant_isolation",
        "audit",
        "retention_policy",
        "deletion_policy",
        "redaction",
        "canary",
        "observability",
        "artifact",
    }
    memory_quality = config["agent_memory_lineage_quality"]
    assert memory_quality["required_operation_types"] == [
        "read",
        "recall",
        "update",
        "write",
    ]
    assert memory_quality["required_policies"] == [
        "audit",
        "canary",
        "deletion",
        "redaction",
        "retention",
        "tenant_isolation",
    ]
    assert set(config["required_retrieval_memory_trace"]) >= {
        "retrieval_memory",
        "trace",
        "query",
        "document",
        "citation",
        "attribution",
        "freshness",
        "memory_write",
    }
    assert config["metric_weights"]["agent_memory_lineage_coverage"] == pytest.approx(4.0)
    assert config["metric_weights"]["agent_memory_lineage_quality"] == pytest.approx(4.0)
    assert config["metric_weights"]["retrieval_memory_attribution"] == pytest.approx(4.0)
    metrics = result["summary"]["metric_averages"]
    assert metrics["agent_memory_lineage_coverage"] == pytest.approx(1.0)
    assert metrics["agent_memory_lineage_quality"] == pytest.approx(1.0)
    assert metrics["retrieval_memory_attribution"] == pytest.approx(1.0)
    state = result["report"]["results"][0]["metadata"]["environment_state"]
    memory = state["framework_memory"]
    assert memory["operation_types"] == ["read", "recall", "update", "write"]
    assert memory["checkpoint_count"] == 1
    assert state["retrieval_memory"]["citations"][0]["doc_ids"] == [
        "refund_policy_doc"
    ]
    assert state["agent_memory_lineage"]["stores"][0]["id"] == "langgraph_store"
    event_types = set(
        state["framework_runtime"]["invocations"][0]["output"]["event_types"]
    )
    assert {
        "framework_memory_operation",
        "framework_memory_checkpoint",
        "framework_memory_retrieval",
        "framework_memory_record",
    } <= event_types


def test_sdk_framework_adapter_browser_cua_trace_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_browser_cua_trace.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_browser_cua_trace",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-browser-cua-trace.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    manifest = result["framework_adapter_browser_cua_trace_manifest"]
    assert manifest["agent"]["framework"] == "browser_use"
    assert manifest["agent"]["method"] == "execute_task"
    assert manifest["agent"]["input_mode"] == "dict"
    assert manifest["agent"]["trace_runtime"] is True
    assert manifest["simulation"]["modality"] == "cua"
    assert manifest["metadata"]["promoted_from_framework_adapter_probe"] is True
    assert manifest["metadata"]["framework_adapter_discovery_used"] is True
    assert manifest["metadata"]["framework_adapter_discovery_status"] == "passed"
    assert manifest["agent"]["metadata"]["adapter_candidate_source"] == "discovery"
    proof = manifest["agent"]["metadata"]["framework_adapter_probe_proof"]
    assert proof["status"] == "passed"
    assert proof["failed_check_ids"] == []
    assert proof["framework"] == "browser_use"
    assert proof["method"] == "execute_task"
    assert proof["input_mode"] == "dict"
    config = manifest["evaluation"]["agent_report"]["config"]
    runtime_contract = config["framework_runtime_contract"]
    assert runtime_contract["required_state_keys"] == ["browser_cua"]
    assert runtime_contract["required_tools"] == ["browser_click"]
    assert set(runtime_contract["required_signals"]) >= {
        "artifact",
        "browser",
        "event",
        "state",
        "tool",
    }
    assert set(config["required_events"]) >= {
        "browser_snapshot",
        "browser_action",
        "browser_trace",
        "browser_network",
        "browser_runtime",
        "browser_storage",
        "environment_injection",
    }
    assert set(config["required_browser_trace"]) >= {
        "trace",
        "snapshot",
        "dom",
        "screenshot",
        "action",
        "action_replay",
        "coordinate_region",
        "screenshot_diff",
        "storage_state",
        "runtime_event",
        "performance_entry",
        "network",
        "prompt_injection_surface",
        "layout_shift",
        "layout_shift_distribution",
        "browser_mutation",
        "browser_mutation_pack",
        "selector_drift",
    }
    assert config["expected_browser_actions"][0] == {
        "tool": "browser_click",
        "action": "click",
        "selector": "button[data-testid='place-order-safe']",
        "success": True,
        "matched": True,
        "blocked": False,
        "mutation_id": "selector_drift_safe_fallback",
        "mutation_type": "selector_drift",
    }
    assert config["expected_browser_regions"][0] == {
        "name": "place_order_button",
        "selector": "button[data-testid='place-order-safe']",
        "success": True,
    }
    assert config["expected_browser_runtime_events"][0] == {
        "type": "console",
        "message_contains": "checkout action replay complete",
    }
    assert config["expected_browser_screenshot_diffs"] == [{}]
    assert config["expected_browser_perturbations"] == [
        {"id": "layout_shift_distribution", "type": "layout_shift"}
    ]
    assert config["required_browser_mutations"] == ["selector_drift_safe_fallback"]
    assert config["browser_mutation_resilience"]["required_types"] == [
        "selector_drift"
    ]
    assert set(config["browser_mutation_resilience"]["required_mitigations"]) >= {
        "browser_mutations",
        "storage_recheck",
        "runtime_recheck",
        "browser_click",
        "mutation_action",
        "mutation_action_success",
    }
    assert config["allow_stale_browser_screenshot"] is False
    assert config["max_browser_performance_duration_ms"] == pytest.approx(18.0)
    assert config["forbidden_browser_prompt_injection_targets"] == [
        {"id": "promo-injection"}
    ]
    assert config["metric_weights"]["browser_action_safety"] == pytest.approx(4.0)
    assert config["metric_weights"]["browser_action_outcome"] == pytest.approx(4.0)
    assert config["metric_weights"]["browser_grounding_quality"] == pytest.approx(4.0)
    assert config["metric_weights"]["browser_mutation_resilience"] == pytest.approx(4.0)
    assert config["metric_weights"]["browser_trace_coverage"] == pytest.approx(4.0)
    metrics = result["summary"]["metric_averages"]
    assert metrics["framework_adapter_call_contract_quality"] == pytest.approx(1.0)
    assert metrics["framework_adapter_contract_quality"] == pytest.approx(1.0)
    assert metrics["framework_adapter_observed_io_quality"] == pytest.approx(1.0)
    assert metrics["framework_runtime_contract"] == pytest.approx(1.0)
    assert metrics["framework_trace_coverage"] == pytest.approx(1.0)
    assert metrics["tool_selection_accuracy"] == pytest.approx(1.0)
    assert metrics["browser_action_safety"] == pytest.approx(1.0)
    assert metrics["browser_action_outcome"] == pytest.approx(1.0)
    assert metrics["browser_grounding_quality"] == pytest.approx(1.0)
    assert metrics["browser_mutation_resilience"] == pytest.approx(1.0)
    assert metrics["browser_trace_coverage"] == pytest.approx(1.0)
    state = result["report"]["results"][0]["metadata"]["environment_state"]
    runtime = state["framework_runtime"]
    assert runtime["summary"]["framework"] == "browser_use"
    assert runtime["summary"]["methods"] == ["execute_task"]
    assert runtime["summary"]["input_modes"] == ["dict"]
    assert runtime["summary"]["call_styles"] == ["positional"]
    browser = state["browser_cua"]
    assert browser["snapshot_count"] == 2
    assert browser["action_count"] == 1
    assert browser["successful_action_count"] == 1
    assert browser["matched_action_count"] == 1
    assert browser["blocked_action_count"] == 0
    assert browser["screenshot_count"] == 2
    assert browser["region_count"] == 1
    assert browser["prompt_injection_surface_count"] == 1
    assert browser["prompt_injection_touched_count"] == 0
    assert browser["mutation_count"] == 1
    assert browser["layout_shift_present"] is True
    assert browser["storage_present"] is True
    assert browser["tool_names"] == ["browser_click"]
    output = runtime["invocations"][0]["output"]
    assert output["tool_names"] == ["browser_click"]
    assert "browser_cua" in output["state_keys"]
    assert {"screenshot", "trace"} <= set(output["artifact_types"])
    assert {
        "browser_snapshot",
        "browser_action",
        "browser_trace",
        "browser_network",
        "browser_runtime",
        "browser_storage",
        "environment_injection",
    } <= set(output["event_types"])


def test_sdk_framework_adapter_workflow_trace_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_workflow_trace.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_workflow_trace",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-workflow-trace.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    manifest = result["framework_adapter_workflow_trace_manifest"]
    assert manifest["agent"]["framework"] == "langgraph"
    assert manifest["agent"]["method"] == "execute_task"
    assert manifest["agent"]["input_mode"] == "dict"
    assert manifest["agent"]["trace_runtime"] is True
    assert manifest["metadata"]["promoted_from_framework_adapter_probe"] is True
    assert manifest["metadata"]["framework_adapter_discovery_used"] is True
    assert manifest["metadata"]["framework_adapter_discovery_status"] == "passed"
    assert manifest["agent"]["metadata"]["adapter_candidate_source"] == "discovery"
    proof = manifest["agent"]["metadata"]["framework_adapter_probe_proof"]
    assert proof["status"] == "passed"
    assert proof["failed_check_ids"] == []
    assert proof["framework"] == "langgraph"
    assert proof["method"] == "execute_task"
    assert proof["input_mode"] == "dict"
    selected_summary = manifest["agent"]["metadata"][
        "framework_adapter_probe_report_summary"
    ]
    assert selected_summary["call_styles"] == ["positional"]
    assert selected_summary["framework"] == "langgraph"
    assert selected_summary["method"] == "execute_task"
    assert selected_summary["input_mode"] == "dict"
    assert selected_summary["tool_call_count"] == 1
    runtime_contract = manifest["evaluation"]["agent_report"]["config"][
        "framework_runtime_contract"
    ]
    config = manifest["evaluation"]["agent_report"]["config"]
    assert runtime_contract["required_state_keys"] == ["workflow_trace"]
    assert runtime_contract["required_tools"] == ["policy_lookup"]
    assert runtime_contract["required_artifact_types"] == ["trace"]
    assert "workflow" in runtime_contract["required_signals"]
    assert "workflow" in config["required_framework_runtime"]
    assert set(config["required_workflow_trace"]) >= {
        "workflow_trace",
        "trace",
        "graph",
        "node",
        "edge",
        "step",
        "checkpoint",
        "route",
        "interrupt",
        "replay",
        "write",
        "state",
        "tool",
        "tool_call",
        "final_state",
        "topology",
        "framework",
    }
    workflow_quality = config["workflow_trace_quality"]
    assert workflow_quality["min_node_count"] == 4
    assert workflow_quality["min_edge_count"] == 3
    assert workflow_quality["min_step_count"] == 4
    assert workflow_quality["min_checkpoint_count"] == 2
    assert workflow_quality["min_route_decision_count"] == 1
    assert workflow_quality["min_interrupt_count"] == 1
    assert workflow_quality["min_replay_count"] == 1
    assert workflow_quality["min_write_count"] == 1
    assert workflow_quality["min_tool_call_count"] == 1
    assert workflow_quality["required_tools"] == ["policy_lookup"]
    assert set(workflow_quality["required_final_state_keys"]) == {
        "approval",
        "decision",
        "policy_result",
    }
    assert workflow_quality["require_replay"] is True
    assert workflow_quality["require_interrupts"] is True
    assert workflow_quality["require_routes"] is True
    assert workflow_quality["require_topology"] is True
    metric_weights = config["metric_weights"]
    assert metric_weights["workflow_trace_coverage"] == pytest.approx(4.0)
    assert metric_weights["workflow_graph_quality"] == pytest.approx(4.0)
    assert set(manifest["evaluation"]["agent_report"]["config"]["required_events"]) >= {
        "workflow_step",
        "workflow_route",
        "workflow_checkpoint",
        "workflow_interrupt",
        "workflow_replay",
        "workflow_trace",
    }
    state = result["report"]["results"][0]["metadata"]["environment_state"]
    workflow = state["workflow_trace"]
    assert workflow["node_count"] == 4
    assert workflow["edge_count"] == 3
    assert workflow["step_count"] == 4
    assert workflow["checkpoint_count"] == 2
    assert workflow["route_decision_count"] == 1
    assert workflow["interrupt_count"] == 1
    assert workflow["replay_count"] == 1
    assert workflow["tool_names"] == ["policy_lookup"]
    metric_averages = result["summary"]["metric_averages"]
    assert metric_averages["workflow_trace_coverage"] == pytest.approx(1.0)
    assert metric_averages["workflow_graph_quality"] == pytest.approx(1.0)
    assert metric_averages["framework_adapter_call_contract_quality"] == (
        pytest.approx(1.0)
    )
    assert metric_averages["framework_adapter_contract_quality"] == pytest.approx(1.0)
    assert metric_averages["framework_adapter_observed_io_quality"] == (
        pytest.approx(1.0)
    )
    assert metric_averages["framework_runtime_contract"] == pytest.approx(1.0)
    assert metric_averages["framework_trace_coverage"] == pytest.approx(1.0)
    assert metric_averages["tool_selection_accuracy"] == pytest.approx(1.0)
    runtime_summary = state["framework_runtime"]["summary"]
    assert runtime_summary["framework"] == "langgraph"
    assert runtime_summary["methods"] == ["execute_task"]
    assert runtime_summary["input_modes"] == ["dict"]
    assert runtime_summary["call_styles"] == ["positional"]
    assert runtime_summary["tool_call_count"] == 1
    output = state["framework_runtime"]["invocations"][0]["output"]
    assert output["tool_names"] == ["policy_lookup"]
    assert {"trace"} <= set(output["artifact_types"])
    assert {
        "workflow_step",
        "workflow_route",
        "workflow_checkpoint",
        "workflow_interrupt",
        "workflow_replay",
        "workflow_trace",
    } <= set(output["event_types"])


def test_sdk_framework_adapter_lifecycle_trace_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_lifecycle_trace.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_lifecycle_trace",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-lifecycle-trace.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    manifest = result["framework_adapter_lifecycle_trace_manifest"]
    assert manifest["agent"]["method"] == "execute_task"
    config = manifest["evaluation"]["agent_report"]["config"]
    runtime_contract = config["framework_runtime_contract"]
    assert runtime_contract["required_state_keys"] == ["framework_lifecycle_trace"]
    assert runtime_contract["required_tools"] == ["framework_lifecycle_status"]
    assert runtime_contract["required_artifact_types"] == ["trace"]
    assert set(config["required_framework_lifecycle"]) >= {
        "retry",
        "cancellation",
        "resume",
        "cleanup",
        "state_persistence",
        "recovery",
    }
    assert result["summary"]["metric_averages"]["framework_lifecycle_coverage"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"]["framework_lifecycle_quality"] == (
        pytest.approx(1.0)
    )
    state = result["report"]["results"][0]["metadata"]["environment_state"]
    lifecycle = state["framework_lifecycle_trace"]
    summary = lifecycle["summary"]
    assert summary["phase_count"] == 10
    assert summary["retry_count"] == 1
    assert summary["error_count"] == 1
    assert summary["recovered_error_count"] == 1
    assert summary["cancellation_count"] == 1
    assert summary["resume_count"] == 1
    assert summary["cleanup_count"] == 1
    assert summary["terminal_status"] == "completed"
    output = state["framework_runtime"]["invocations"][0]["output"]
    assert output["tool_names"] == ["framework_lifecycle_status"]
    assert {"trace"} <= set(output["artifact_types"])
    assert {
        "framework_lifecycle_phase",
        "framework_lifecycle_trace",
    } <= set(output["event_types"])


def test_sdk_framework_adapter_trace_export_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_trace_export.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_trace_export",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-trace-export.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    manifest = result["framework_adapter_trace_export_manifest"]
    assert manifest["agent"]["method"] == "execute_task"
    config = manifest["evaluation"]["agent_report"]["config"]
    runtime_contract = config["framework_runtime_contract"]
    assert runtime_contract["required_state_keys"] == ["framework_trace"]
    assert runtime_contract["required_tools"] == ["policy_lookup"]
    assert runtime_contract["required_artifact_types"] == ["trace"]
    assert set(config["required_framework_trace"]) >= {
        "framework_trace",
        "model",
        "tool",
        "state",
        "latency",
        "cost",
        "span",
    }
    trace_quality = config["framework_trace_quality"]
    assert trace_quality["framework"] == "langgraph"
    assert trace_quality["min_span_count"] == 3
    assert trace_quality["min_model_span_count"] == 1
    assert trace_quality["min_tool_span_count"] == 1
    assert trace_quality["min_state_span_count"] == 1
    assert trace_quality["min_latency_span_count"] == 3
    assert trace_quality["min_cost_span_count"] == 1
    assert trace_quality["min_tool_count"] == 1
    assert trace_quality["max_error_count"] == 0
    assert trace_quality["require_adapter_conformance"] is True
    assert trace_quality["max_adapter_conformance_findings"] == 0
    assert trace_quality["required_tools"] == ["policy_lookup"]
    assert {"model", "tool", "state", "latency", "cost"} <= set(
        trace_quality["required_signals"]
    )
    assert set(trace_quality["required_spans"]) >= {
        "langgraph checkpoint refund decision",
        "langgraph refund model chat",
        "tool call policy_lookup",
    }
    assert config["metric_weights"]["framework_trace_coverage"] == pytest.approx(4.0)
    assert config["metric_weights"]["framework_trace_quality"] == pytest.approx(4.0)
    assert result["summary"]["metric_averages"]["framework_trace_coverage"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"]["framework_trace_quality"] == (
        pytest.approx(1.0)
    )
    state = result["report"]["results"][0]["metadata"]["environment_state"]
    trace = state["framework_trace"]
    summary = trace["summary"]
    assert summary["span_count"] == 3
    assert summary["tool_names"] == ["policy_lookup"]
    assert summary["model_span_count"] == 1
    assert summary["tool_span_count"] == 1
    assert summary["state_span_count"] == 1
    assert trace["adapter_conformance"]["passed"] is True
    output = state["framework_runtime"]["invocations"][0]["output"]
    assert output["tool_names"] == ["policy_lookup"]
    assert {"trace"} <= set(output["artifact_types"])
    assert {
        "framework_trace_span",
        "framework_trace",
    } <= set(output["event_types"])


def test_sdk_framework_adapter_orchestration_trace_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_orchestration_trace.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_orchestration_trace",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-orchestration-trace.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    manifest = result["framework_adapter_orchestration_trace_manifest"]
    assert manifest["agent"]["framework"] == "langgraph"
    assert manifest["agent"]["method"] == "execute_task"
    assert manifest["agent"]["input_mode"] == "dict"
    assert manifest["agent"]["trace_runtime"] is True
    assert manifest["metadata"]["promoted_from_framework_adapter_probe"] is True
    assert manifest["metadata"]["framework_adapter_discovery_used"] is True
    assert manifest["metadata"]["framework_adapter_discovery_status"] == "passed"
    assert manifest["agent"]["metadata"]["adapter_candidate_source"] == "discovery"
    proof = manifest["agent"]["metadata"]["framework_adapter_probe_proof"]
    assert proof["status"] == "passed"
    assert proof["failed_check_ids"] == []
    assert proof["framework"] == "langgraph"
    assert proof["method"] == "execute_task"
    assert proof["input_mode"] == "dict"
    selected_summary = manifest["agent"]["metadata"][
        "framework_adapter_probe_report_summary"
    ]
    assert selected_summary["call_styles"] == ["positional"]
    assert selected_summary["framework"] == "langgraph"
    assert selected_summary["method"] == "execute_task"
    assert selected_summary["input_mode"] == "dict"
    assert selected_summary["tool_call_count"] == 2
    config = manifest["evaluation"]["agent_report"]["config"]
    runtime_contract = config["framework_runtime_contract"]
    assert runtime_contract["required_state_keys"] == ["orchestration_trace"]
    assert runtime_contract["required_tools"] == ["policy_lookup"]
    assert runtime_contract["required_artifact_types"] == ["trace"]
    assert set(config["required_orchestration_trace"]) >= {
        "orchestration_trace",
        "trace",
        "step",
        "node",
        "route",
        "agent",
        "spawn",
        "delegate",
        "handoff",
        "communicate",
        "aggregate",
        "stop",
        "retry",
        "recovered",
        "latency",
        "cost",
        "tool",
        "state",
    }
    assert config["metric_weights"]["orchestration_trace_coverage"] == pytest.approx(4.0)
    assert config["metric_weights"]["orchestration_flow_quality"] == pytest.approx(4.0)
    assert result["summary"]["metric_averages"]["orchestration_trace_coverage"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"]["orchestration_flow_quality"] == (
        pytest.approx(1.0)
    )
    metric_averages = result["summary"]["metric_averages"]
    assert metric_averages["framework_adapter_call_contract_quality"] == (
        pytest.approx(1.0)
    )
    assert metric_averages["framework_adapter_contract_quality"] == pytest.approx(1.0)
    assert metric_averages["framework_adapter_observed_io_quality"] == (
        pytest.approx(1.0)
    )
    assert metric_averages["framework_runtime_contract"] == pytest.approx(1.0)
    assert metric_averages["framework_trace_coverage"] == pytest.approx(1.0)
    assert metric_averages["tool_selection_accuracy"] == pytest.approx(1.0)
    state = result["report"]["results"][0]["metadata"]["environment_state"]
    runtime_summary = state["framework_runtime"]["summary"]
    assert runtime_summary["framework"] == "langgraph"
    assert runtime_summary["methods"] == ["execute_task"]
    assert runtime_summary["input_modes"] == ["dict"]
    assert runtime_summary["call_styles"] == ["positional"]
    assert runtime_summary["tool_call_count"] == 2
    trace = state["orchestration_trace"]
    summary = trace["summary"]
    assert summary["node_count"] == 4
    assert summary["edge_count"] == 3
    assert summary["step_count"] == 6
    assert summary["spawn_count"] == 1
    assert summary["delegation_count"] == 2
    assert summary["communication_count"] == 2
    assert summary["aggregation_count"] == 2
    assert summary["stop_count"] == 1
    assert summary["failure_count"] == 1
    assert summary["retry_count"] == 1
    assert summary["recovered_failures"] == 1
    assert summary["terminal_status"] == "success"
    output = state["framework_runtime"]["invocations"][0]["output"]
    assert output["tool_names"] == ["policy_lookup"]
    assert {"trace"} <= set(output["artifact_types"])
    assert {
        "orchestration_step",
        "orchestration_trace",
    } <= set(output["event_types"])


def test_sdk_framework_adapter_mcp_tool_session_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_mcp_tool_session.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_mcp_tool_session",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-mcp-tool-session.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    manifest = result["framework_adapter_mcp_tool_session_manifest"]
    assert manifest["agent"]["method"] == "execute_task"
    config = manifest["evaluation"]["agent_report"]["config"]
    runtime_contract = config["framework_runtime_contract"]
    assert runtime_contract["required_state_keys"] == ["mcp_tool_session"]
    assert runtime_contract["required_tools"] == [
        "refund_policy_lookup",
        "refund_status",
    ]
    assert runtime_contract["required_artifact_types"] == ["trace"]
    assert set(config["required_mcp_tool_session"]) >= {
        "mcp_tool_session",
        "trace",
        "server",
        "session",
        "tool",
        "tool_schema",
        "resource",
        "tool_call",
        "tool_result",
    }
    assert config["mcp_tool_session_quality"]["required_tools"] == [
        "refund_policy_lookup",
        "refund_status",
    ]
    assert config["mcp_tool_session_quality"]["required_servers"] == ["refund-tools"]
    assert config["metric_weights"]["mcp_tool_session_coverage"] == pytest.approx(4.0)
    assert config["metric_weights"]["mcp_tool_session_quality"] == pytest.approx(4.0)
    assert result["summary"]["metric_averages"]["mcp_tool_session_coverage"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"]["mcp_tool_session_quality"] == (
        pytest.approx(1.0)
    )
    assert {
        "mcp_server",
        "mcp_tool_schema",
        "mcp_resource",
        "mcp_tool_call",
        "mcp_tool_result",
        "mcp_tool_session",
    } <= set(config["required_events"])
    state = result["report"]["results"][0]["metadata"]["environment_state"]
    summary = state["mcp_tool_session"]["summary"]
    assert summary["schema_count"] == 2
    assert summary["resource_count"] == 1
    assert summary["call_count"] == 2
    assert summary["result_count"] == 2
    assert summary["tool_names"] == ["refund_policy_lookup", "refund_status"]
    output = state["framework_runtime"]["invocations"][0]["output"]
    assert output["tool_names"] == ["refund_policy_lookup", "refund_status"]
    assert {"trace"} <= set(output["artifact_types"])
    assert {
        "mcp_server",
        "mcp_tool_schema",
        "mcp_resource",
        "mcp_tool_call",
        "mcp_tool_result",
        "mcp_tool_session",
    } <= set(output["event_types"])


def test_sdk_framework_adapter_a2a_protocol_trace_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_a2a_protocol_trace.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_a2a_protocol_trace",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-a2a-protocol-trace.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    manifest = result["framework_adapter_a2a_protocol_trace_manifest"]
    assert manifest["agent"]["method"] == "send_message"
    config = manifest["evaluation"]["agent_report"]["config"]
    runtime_contract = config["framework_runtime_contract"]
    assert runtime_contract["required_state_keys"] == ["a2a_protocol_trace"]
    assert set(runtime_contract["required_artifact_types"]) == {"trace", "json"}
    assert set(config["required_a2a_protocol"]) >= {
        "a2a_protocol_trace",
        "trace",
        "agent_card",
        "skill",
        "message",
        "task",
        "artifact",
        "protocol_event",
        "part",
        "text_part",
        "data_part",
        "status_update",
        "artifact_update",
        "terminal_task",
        "role",
        "state",
        "task_id",
        "context",
    }
    assert config["a2a_protocol_quality"]["required_agents"] == [
        "refund-review-agent"
    ]
    assert config["a2a_protocol_quality"]["required_skills"] == ["refund_review"]
    assert config["a2a_protocol_quality"]["required_roles"] == ["agent", "user"]
    assert config["a2a_protocol_quality"]["required_states"] == ["completed"]
    assert config["metric_weights"]["a2a_protocol_coverage"] == pytest.approx(4.0)
    assert config["metric_weights"]["a2a_protocol_quality"] == pytest.approx(4.0)
    assert result["summary"]["metric_averages"]["a2a_protocol_coverage"] == (
        pytest.approx(1.0)
    )
    assert result["summary"]["metric_averages"]["a2a_protocol_quality"] == (
        pytest.approx(1.0)
    )
    assert {
        "a2a_agent_card",
        "a2a_message_send",
        "a2a_task_status",
        "a2a_task_artifact",
        "a2a_artifact",
        "a2a_protocol_trace",
    } <= set(config["required_events"])
    state = result["report"]["results"][0]["metadata"]["environment_state"]
    summary = state["a2a_protocol_trace"]["summary"]
    assert summary["agent_card_count"] == 1
    assert summary["message_count"] == 3
    assert summary["task_count"] == 1
    assert summary["artifact_count"] == 1
    assert summary["status_update_count"] == 3
    assert summary["skill_names"] == ["refund_review"]
    output = state["framework_runtime"]["invocations"][0]["output"]
    assert {"trace", "json"} <= set(output["artifact_types"])
    assert {
        "a2a_agent_card",
        "a2a_message_send",
        "a2a_task_status",
        "a2a_task_artifact",
        "a2a_artifact",
        "a2a_protocol_trace",
    } <= set(output["event_types"])


def test_sdk_framework_adapter_agent_control_plane_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_framework_adapter_agent_control_plane.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_framework_adapter_agent_control_plane",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-framework-adapter-agent-control-plane.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    manifest = result["framework_adapter_agent_control_plane_manifest"]
    assert manifest["agent"]["framework"] == "agent_learning_kit"
    assert manifest["agent"]["method"] == "execute_task"
    assert manifest["agent"]["input_mode"] == "dict"
    assert manifest["agent"]["metadata"]["adapter_candidate_source"] == "discovery"
    assert manifest["agent"]["metadata"]["framework_adapter_discovery_used"] is True
    config = manifest["evaluation"]["agent_report"]["config"]
    runtime_contract = config["framework_runtime_contract"]
    assert runtime_contract["required_state_keys"] == [
        "agent_control_plane",
        "agent_trust_boundary_model",
        "framework_trace",
    ]
    assert "control_plane" in runtime_contract["required_signals"]
    assert {
        *trinity.V1_AGENT_CONTROL_PLANE_REQUIRED_EVENTS,
        "framework_runtime",
        "framework_trace",
        "framework_trace_span",
    } <= set(config["required_events"])

    metric_averages = result["summary"]["metric_averages"]
    for metric in (
        "framework_adapter_call_contract_quality",
        "framework_adapter_contract_quality",
        "framework_adapter_observed_io_quality",
        "framework_runtime_contract",
        "framework_trace_coverage",
        "agent_trust_boundary_coverage",
        "agent_trust_boundary_quality",
        "agent_control_plane_coverage",
        "agent_control_plane_quality",
        "tool_selection_accuracy",
    ):
        assert metric_averages[metric] == pytest.approx(1.0)

    state = result["report"]["results"][0]["metadata"]["environment_state"]
    assert {
        "agent_control_plane",
        "agent_trust_boundary_model",
        "framework_runtime",
        "framework_trace",
    } <= set(state)
    runtime = state["framework_runtime"]
    assert "control_plane" in runtime["signals"]
    output = runtime["invocations"][0]["output"]
    assert {
        "agent_control_plane",
        "agent_trust_boundary_model",
        "framework_trace",
    } <= set(output["state_keys"])
    assert {"trace"} <= set(output["artifact_types"])

    trust_summary = state["agent_trust_boundary_model"]["summary"]
    assert trust_summary["control_count"] == 11
    assert trust_summary["required_control_rate"] == pytest.approx(1.0)
    assert trust_summary["high_risk_unmitigated_count"] == 0
    assert trust_summary["gaps"] == []
    for flag in trinity.V1_AGENT_TRUST_BOUNDARY_REQUIRED_FLAGS:
        assert trust_summary[flag] is True

    control_summary = state["agent_control_plane"]["summary"]
    assert control_summary["control_count"] == 11
    assert control_summary["required_control_rate"] == pytest.approx(1.0)
    assert control_summary["approval_required_action_count"] >= 2
    assert control_summary["blocked_action_count"] >= 1
    assert control_summary["rolled_back_action_count"] >= 1
    assert control_summary["contained_incident_count"] >= 1
    assert control_summary["within_budget_count"] >= 3
    assert control_summary["exceeded_budget_count"] == 0
    assert control_summary["high_risk_uncontained_count"] == 0
    assert control_summary["gaps"] == []
    for flag in trinity.V1_AGENT_CONTROL_PLANE_REQUIRED_FLAGS:
        assert control_summary[flag] is True


def test_sdk_memory_layer_probe_optimization_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_memory_layer_probe_optimization.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_memory_layer_probe_optimization",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-memory-layer-probe-optimization.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        output_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    assert result["summary"]["metric_averages"][
        "agent_memory_lineage_quality"
    ] == pytest.approx(1.0)
    assert manifest["metadata"]["promoted_from_memory_layer_probe"] is True
    assert manifest["metadata"]["memory_layer_probe_proof_status"] == "passed"
    assert [env["type"] for env in manifest["simulation"]["environments"]] == [
        "retrieval_memory",
        "agent_memory_lineage",
    ]
    assert manifest["simulation"]["environments"][0]["data"]["documents"][0][
        "id"
    ] == "doc_refund_2026"


def test_sdk_multi_agent_room_probe_optimization_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_multi_agent_room_probe_optimization.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_multi_agent_room_probe_optimization",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-multi-agent-room-probe-optimization.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        output_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    assert result["summary"]["metric_averages"][
        "multi_agent_coordination_quality"
    ] == pytest.approx(1.0)
    assert manifest["metadata"]["promoted_from_multi_agent_room_probe"] is True
    assert manifest["metadata"]["multi_agent_room_probe_proof_status"] == "passed"
    assert [env["type"] for env in manifest["simulation"]["environments"]] == [
        "multi_agent_room"
    ]
    assert manifest["simulation"]["environments"][0]["data"]["expected_reconciliation"][
        "accepted_source"
    ] == "critic"


def test_sdk_orchestration_stack_probe_optimization_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_orchestration_stack_probe_optimization.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_orchestration_stack_probe_optimization",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-orchestration-stack-probe-optimization.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        output_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    assert result["summary"]["metric_averages"][
        "world_contract_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "agent_memory_lineage_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "multi_agent_coordination_quality"
    ] == pytest.approx(1.0)
    assert manifest["metadata"]["promoted_from_orchestration_stack_probe"] is True
    assert manifest["metadata"]["orchestration_stack_probe_proof_status"] == "passed"
    assert [env["type"] for env in manifest["simulation"]["environments"]] == [
        "world_contract",
        "framework_trace",
        "retrieval_memory",
        "agent_memory_lineage",
        "multi_agent_room",
    ]
    assert manifest["simulation"]["environments"][0]["data"]["transitions"][0][
        "id"
    ] == "approve_refund"


def test_sdk_evaluation_hook_probe_optimization_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_evaluation_hook_probe_optimization.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_evaluation_hook_probe_optimization",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-evaluation-hook-probe-optimization.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        output_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    assert result["summary"]["metric_averages"][
        "external_task_quality"
    ] == pytest.approx(1.0)
    assert manifest["required_env"] == []
    assert manifest["metadata"]["promoted_from_evaluation_hook_probe"] is True
    assert manifest["metadata"]["evaluation_hook_probe_proof_status"] == "passed"
    hook = manifest["evaluation"]["agent_report"]["config"]["evaluation_hooks"][0]
    assert hook["endpoint"].startswith("http://127.0.0.1:")
    assert hook["metric_name"] == "external_task_quality"


def test_sdk_trinity_stack_probe_optimization_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_trinity_stack_probe_optimization.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_trinity_stack_probe_optimization",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-trinity-stack-probe-optimization.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        output_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    metrics = result["summary"]["metric_averages"]
    assert metrics["external_task_quality"] == pytest.approx(1.0)
    assert metrics["world_contract_quality"] == pytest.approx(1.0)
    assert metrics["agent_memory_lineage_quality"] == pytest.approx(1.0)
    assert metrics["multi_agent_coordination_quality"] == pytest.approx(1.0)
    assert manifest["required_env"] == []
    assert manifest["metadata"]["promoted_from_trinity_stack_probe"] is True
    assert manifest["metadata"]["trinity_stack_probe_proof_status"] == "passed"
    hook = manifest["evaluation"]["agent_report"]["config"]["evaluation_hooks"][0]
    assert hook["endpoint"].startswith("http://127.0.0.1:")
    assert [env["type"] for env in manifest["simulation"]["environments"]] == [
        "world_contract",
        "framework_trace",
        "retrieval_memory",
        "agent_memory_lineage",
        "multi_agent_room",
    ]


def test_sdk_realtime_stack_probe_optimization_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_realtime_stack_probe_optimization.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_realtime_stack_probe_optimization",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-realtime-stack-probe-optimization.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        output_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    assert result["summary"]["metric_averages"][
        "streaming_interaction_quality"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "voice_timing_distribution_quality"
    ] == pytest.approx(1.0)
    assert manifest["metadata"]["promoted_from_realtime_stack_probe"] is True
    assert manifest["metadata"]["realtime_stack_probe_proof_status"] == "passed"
    assert [env["type"] for env in manifest["simulation"]["environments"]] == [
        "voice",
        "streaming_trace",
    ]
    assert manifest["simulation"]["environments"][0]["data"]["sample_rate_hz"] == 16000
    assert (
        manifest["simulation"]["environments"][1]["data"]["state"]["route"]
        == "support"
    )


def test_sdk_browser_cua_probe_optimization_example_runs(tmp_path):
    example_path = EXAMPLES / "sdk_browser_cua_probe_optimization.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_browser_cua_probe_optimization",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "sdk-browser-cua-probe-optimization.json"
    result = module.run(output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        output_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )

    assert saved == result
    assert result["kind"] == "agent-learning.run.v1"
    assert result["status"] == "passed"
    assert result["summary"]["metric_averages"][
        "browser_action_outcome"
    ] == pytest.approx(1.0)
    assert result["summary"]["metric_averages"][
        "browser_trace_coverage"
    ] == pytest.approx(1.0)
    assert manifest["metadata"]["promoted_from_browser_cua_probe"] is True
    assert manifest["metadata"]["browser_cua_probe_proof_status"] == "passed"
    assert [env["type"] for env in manifest["simulation"]["environments"]] == [
        "browser_cua"
    ]
    browser = manifest["simulation"]["environments"][0]["data"]
    assert browser["metadata"]["trace_provider"] == "local_browser_cua"
    assert len(browser["mutation_pack"]["mutations"]) == 2


def test_world_framework_memory_optimization_example_runs_evidence_gates(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_WORLD_FRAMEWORK_OPT_EXAMPLE_KEY",
        "real-local-world-framework-key",
    )

    output_path = tmp_path / "world-framework-memory.json"
    junit_path = tmp_path / "world-framework-memory.junit.xml"
    sarif_path = tmp_path / "world-framework-memory.sarif.json"
    markdown_path = tmp_path / "world-framework-memory.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "world_framework_memory_optimization.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.optimization.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] >= 0.84
    assert "simulation.environments" in payload["summary"]["search_paths"]

    env_types = [
        environment["type"]
        for environment in payload["optimization"]["best_config"]["simulation"][
            "environments"
        ]
    ]
    assert env_types == [
        "world_orchestration_replay",
        "framework_trace",
        "retrieval_memory",
        "agent_memory_lineage",
        "multi_agent_room",
    ]

    best_history = max(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    metrics = best_history["metrics"]
    for metric in (
        "orchestration_flow_quality",
        "world_contract_quality",
        "retrieval_context_quality",
        "agent_memory_lineage_quality",
        "multi_agent_coordination_quality",
    ):
        assert metrics[metric] == pytest.approx(1.0)

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    assert json.loads(sarif_path.read_text(encoding="utf-8"))["version"] == "2.1.0"
    assert "world-framework-memory-optimization" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_multi_framework_simulation_suite_runs_framework_adapters(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY",
        "real-local-multi-framework-key",
    )

    output_path = tmp_path / "multi-framework-suite.json"
    junit_path = tmp_path / "multi-framework-suite.junit.xml"
    sarif_path = tmp_path / "multi-framework-suite.sarif.json"
    markdown_path = tmp_path / "multi-framework-suite.md"

    exit_code = main([
        "suite",
        str(EXAMPLES / "multi_framework_simulation_suite.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.suite.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["commands"] == {"run": 10}
    assert payload["summary"]["score"] == pytest.approx(1.0)

    expected = {
        "langchain-runnable": ("langchain", "ainvoke", "dict", "text"),
        "langgraph-state-graph": ("langgraph", "ainvoke", "dict", "text"),
        "llamaindex-chat-engine": ("llamaindex", "achat", "text", "text"),
        "openai-agents-runner": ("openai_agents", "run", "text", "text"),
        "autogen-agent-chat": ("autogen", "run", "text", "text"),
        "crewai-crew": ("crewai", "kickoff", "dict", "text"),
        "pydantic-ai-agent": ("pydantic_ai", "run", "text", "text"),
        "pipecat-voice-pipeline": ("pipecat", "process", "dict", "voice"),
        "livekit-realtime-agent": ("livekit", "respond", "text", "voice"),
        "custom-refund-orchestrator": (
            "custom_refund_orchestrator",
            "execute_task",
            "dict",
            "text",
        ),
    }
    assert set(expected) == {child["id"] for child in payload["children"]}
    for child in payload["children"]:
        framework, method, input_mode, modality = expected[child["id"]]
        assert child["kind"] == "agent-learning.run.v1"
        assert child["status"] == "passed"
        case = child["result"]["report"]["results"][0]
        state = case["metadata"]["environment_state"]
        runtime = state["framework_runtime"]
        summary = runtime["summary"]
        assert runtime["framework"] == framework
        assert runtime["modality"] == modality
        assert summary["framework"] == framework
        assert summary["methods"] == [method]
        assert summary["input_modes"] == [input_mode]
        assert summary["tool_call_count"] == 1
        assert state["framework_trace"]["adapter_conformance"]["passed"] is True
        assistant_messages = [
            message for message in case["messages"] if message["role"] == "assistant"
        ]
        assert "framework_trace_status" in {
            call["name"]
            for message in assistant_messages
            for call in message.get("tool_calls", [])
        }
        assert "framework_status" in {
            message.get("tool_call_id")
            for message in case["messages"]
            if message["role"] == "tool"
        }

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "multi-framework-simulation-suite" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_custom_framework_optimization_example_runs_adapter_search(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_CUSTOM_FRAMEWORK_OPT_EXAMPLE_KEY",
        "real-local-custom-framework-opt-key",
    )

    output_path = tmp_path / "custom-framework-optimization.json"
    optimization_report_path = tmp_path / "custom-framework-optimization-result-report.json"
    optimization_report_markdown_path = (
        tmp_path / "custom-framework-optimization-result-report.md"
    )
    promotion_path = tmp_path / "custom-framework-optimization-promotion.json"
    manifest_path = tmp_path / "custom-framework-optimization-regression.json"
    report_path = tmp_path / "custom-framework-optimization-report.json"
    report_markdown_path = tmp_path / "custom-framework-optimization-report.md"
    replay_path = tmp_path / "custom-framework-optimization-replay.json"
    junit_path = tmp_path / "custom-framework-optimization.junit.xml"
    sarif_path = tmp_path / "custom-framework-optimization.sarif.json"
    markdown_path = tmp_path / "custom-framework-optimization.md"
    replay_junit_path = tmp_path / "custom-framework-optimization-replay.junit.xml"
    replay_sarif_path = tmp_path / "custom-framework-optimization-replay.sarif.json"
    replay_markdown_path = tmp_path / "custom-framework-optimization-replay.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "custom_framework_optimization.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.optimization.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] >= 0.95
    assert payload["summary"]["evaluation_score"] == pytest.approx(1.0)
    assert "agent" in payload["summary"]["search_paths"]
    assert payload["optimization"]["source_manifest_path"] == str(
        EXAMPLES / "custom_framework_optimization.json"
    )
    source_manifest = payload["optimization"]["source_manifest"]
    assert "optimization" not in source_manifest
    assert source_manifest["agent"]["method"] == "run"

    best_agent = payload["optimization"]["best_config"]["agent"]
    assert best_agent["framework"] == "custom_refund_orchestrator"
    assert best_agent["method"] == "execute_task"
    assert best_agent["input_mode"] == "dict"

    best_history = max(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["patch"].keys() == {"agent"}
    assert best_history["metrics"]["framework_runtime_contract"] == pytest.approx(1.0)
    assert best_history["metrics"]["framework_runtime_coverage"] == pytest.approx(1.0)
    assert best_history["metrics"]["framework_trace_coverage"] == pytest.approx(1.0)

    weakest_history = min(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert weakest_history["metrics"]["framework_runtime_contract"] < 1.0

    case = best_history["report"]["results"][0]
    state = case["metadata"]["environment_state"]
    runtime = state["framework_runtime"]
    assert runtime["framework"] == "custom_refund_orchestrator"
    assert runtime["summary"]["methods"] == ["execute_task"]
    assert runtime["summary"]["input_modes"] == ["dict"]
    assert runtime["summary"]["tool_call_count"] == 1
    assert state["framework_trace"]["adapter_conformance"]["passed"] is True

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "custom-framework-adapter-optimization" in markdown_path.read_text(
        encoding="utf-8"
    )

    exit_code = main([
        "report",
        str(output_path),
        "--output",
        str(optimization_report_path),
        "--markdown",
        str(optimization_report_markdown_path),
    ])
    assert exit_code == 0

    optimization_report = json.loads(
        optimization_report_path.read_text(encoding="utf-8")
    )
    diagnosis = optimization_report["report"]["harness_diagnosis"]
    rollout_plan = diagnosis["retrospective_rollout_plan"]
    assert rollout_plan["kind"] == "retrospective_harness_rollout_plan"
    assert rollout_plan["method"] == "evidence_calibrated_candidate_lineage"
    assert rollout_plan["selected_candidate_id"] == payload["summary"][
        "best_candidate_id"
    ]
    assert rollout_plan["candidate_count"] == len(payload["optimization"]["history"])
    assert "framework_runtime_contract" in rollout_plan["weak_metric_names"]
    assert {"execution", "observability", "verification"} <= set(
        rollout_plan["target_layers"]
    )
    selected_lineage = next(
        item
        for item in rollout_plan["candidate_lineage"]
        if item["selected"]
    )
    assert selected_lineage["candidate_id"] == payload["summary"]["best_candidate_id"]
    assert selected_lineage["score_delta_from_seed"] > 0
    assert "agent.method" in selected_lineage["patch_paths"]
    execution_frontier = next(
        item
        for item in rollout_plan["repair_frontier"]
        if item["layer"] == "execution"
    )
    assert execution_frontier["status"] == "needs_attention"
    assert "framework_runtime_contract" in execution_frontier["weak_metric_names"]
    assert {step["id"] for step in rollout_plan["rollout_steps"]} == {
        "replay_selected_candidate",
        "repair_weak_layers",
        "promote_or_hold",
    }
    optimization_report_markdown = optimization_report_markdown_path.read_text(
        encoding="utf-8"
    )
    assert "### Retrospective Rollout Plan" in optimization_report_markdown
    assert "### Candidate Lineage" in optimization_report_markdown
    assert "### Repair Frontier" in optimization_report_markdown
    assert "evidence_calibrated_candidate_lineage" in optimization_report_markdown

    exit_code = main([
        "promote-to-regression",
        str(output_path),
        "--output",
        str(promotion_path),
        "--manifest",
        str(manifest_path),
        "--min-level",
        "note",
        "--max-findings",
        "1",
        "--required-env",
        "AGENT_LEARNING_CUSTOM_FRAMEWORK_REGRESSION_KEY",
    ])
    assert exit_code == 0

    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    assert promotion["status"] == "passed"
    assert promotion["summary"]["promotion_kind"] == "optimized_manifest"
    assert promotion["summary"]["promoted_finding_count"] == 0
    assert promotion["summary"]["promoted_manifest_count"] == 1
    assert promotion["summary"]["best_candidate_id"] == payload["summary"][
        "best_candidate_id"
    ]
    assert "agent.method" in promotion["summary"]["search_paths"]
    assert promotion["summary"]["has_optimizer_trace"] is True

    promoted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    promoted_agent = promoted_manifest["agent"]
    assert promoted_agent["method"] == "execute_task"
    assert promoted_agent["input_mode"] == "dict"
    assert promoted_agent["target"].endswith(
        "framework_shims.py:build_custom_refund_orchestrator"
    )
    assert Path(promoted_agent["target"].split(":", 1)[0]).is_absolute()
    assert promoted_manifest["required_env"] == [
        "AGENT_LEARNING_CUSTOM_FRAMEWORK_REGRESSION_KEY"
    ]
    environment_types = {
        environment["type"]
        for environment in promoted_manifest["simulation"]["environments"]
    }
    assert {"framework_trace", "optimizer_trace"} <= environment_types
    assert promoted_manifest["metadata"]["regression"]["promotion_kind"] == (
        "optimized_manifest"
    )

    exit_code = main([
        "report",
        str(promotion_path),
        "--output",
        str(report_path),
        "--markdown",
        str(report_markdown_path),
    ])
    assert exit_code == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert "optimization_replay" in report["summary"]["sections"]
    assert "harness_diagnosis" in report["summary"]["sections"]
    replay_card = report["report"]["optimizer_replay"]
    diagnosis = report["report"]["harness_diagnosis"]
    assert diagnosis["kind"] == "harness_layer_diagnosis"
    assert "observability" in diagnosis["primary_layers"]
    assert {
        "https://arxiv.org/abs/2606.06324",
        "https://arxiv.org/abs/2606.05922",
    } <= set(diagnosis["research_sources"])
    assert replay_card["kind"] == "promotion_manifest"
    assert replay_card["promotion_kind"] == "optimized_manifest"
    assert replay_card["source"]["status"] == "passed"
    assert replay_card["best_candidate_id"] == payload["summary"]["best_candidate_id"]
    assert "agent.method" in replay_card["search_paths"]
    assert "optimizer_trace" in replay_card["environment_types"]
    assert replay_card["has_optimizer_trace"] is True
    assert replay_card["promoted_manifest"]["agent"]["method"] == "execute_task"
    assert replay_card["promoted_manifest"]["agent"]["input_mode"] == "dict"
    assert replay_card["artifacts"]["promoted_manifest"]["agent"]["method"] == (
        "execute_task"
    )
    action_ids = {action["id"] for action in replay_card["actions"]}
    assert {
        "report_artifact",
        "recreate_promotion",
        "replay_promoted_manifest",
        "export_promoted_manifest",
    } <= action_ids
    replay_action = next(
        action
        for action in replay_card["actions"]
        if action["id"] == "replay_promoted_manifest"
    )
    assert replay_action["command_args"][:3] == [
        "agent-learn",
        "replay",
        "{{manifest_path}}",
    ]
    export_action = next(
        action
        for action in replay_card["actions"]
        if action["id"] == "export_promoted_manifest"
    )
    assert export_action["kind"] == "download"
    assert export_action["artifact_ref"] == (
        "report.optimizer_replay.artifacts.promoted_manifest"
    )
    report_markdown = report_markdown_path.read_text(encoding="utf-8")
    assert "## Optimization Replay" in report_markdown
    assert "## Harness Diagnosis" in report_markdown
    assert "### Diagnosis Actions" in report_markdown
    assert "replay_diagnosed_regression" in report_markdown
    assert "Promotion kind" in report_markdown
    assert "optimized_manifest" in report_markdown
    assert "### Promoted Manifest" in report_markdown
    assert "agent.method" in report_markdown
    assert "execute_task" in report_markdown
    assert "optimizer_trace" in report_markdown

    monkeypatch.setenv(
        "AGENT_LEARNING_CUSTOM_FRAMEWORK_REGRESSION_KEY",
        "real-local-custom-framework-regression-key",
    )
    exit_code = main([
        "replay",
        str(manifest_path),
        "--output",
        str(replay_path),
        "--junit",
        str(replay_junit_path),
        "--sarif",
        str(replay_sarif_path),
        "--markdown",
        str(replay_markdown_path),
    ])
    assert exit_code == 0

    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    assert replay["status"] == "passed"
    assert replay["summary"]["replay_pass_rate"] == pytest.approx(1.0)
    child = replay["replay"]["manifests"][0]
    assert child["command"] == "run"
    assert child["status"] == "passed"
    replay_metrics = child["summary"]["metric_averages"]
    assert replay_metrics["framework_runtime_contract"] == pytest.approx(1.0)
    assert replay_metrics["framework_runtime_coverage"] == pytest.approx(1.0)
    assert replay_metrics["framework_trace_coverage"] == pytest.approx(1.0)

    assert "failures=\"0\"" in replay_junit_path.read_text(encoding="utf-8")
    replay_sarif = json.loads(replay_sarif_path.read_text(encoding="utf-8"))
    assert replay_sarif["version"] == "2.1.0"
    assert not [
        result
        for result in replay_sarif["runs"][0]["results"]
        if result.get("level") == "error"
    ]
    replay_markdown = replay_markdown_path.read_text(encoding="utf-8")
    assert "custom-framework-optimization-regression" in replay_markdown
    assert "### Replay Metrics" in replay_markdown
    assert "framework_runtime_contract" in replay_markdown

    replay_report_path = tmp_path / "custom-framework-optimization-replay-report.json"
    exit_code = main([
        "report",
        str(replay_path),
        "--output",
        str(replay_report_path),
    ])
    assert exit_code == 0

    replay_report = json.loads(replay_report_path.read_text(encoding="utf-8"))
    replay_report_card = replay_report["report"]["replay"]
    diagnosis_card = replay_report["report"]["harness_diagnosis"]
    assert diagnosis_card["kind"] == "harness_layer_diagnosis"
    assert {"observability", "verification"} <= {
        layer["layer"]
        for layer in diagnosis_card["layers"]
    }
    assert {
        "report_harness_diagnosis",
        "rerun_diagnosed_replay",
    } <= {action["id"] for action in diagnosis_card["actions"]}
    assert replay_report_card["kind"] == "replay_metrics"
    assert replay_report_card["manifest_count"] == 1
    assert replay_report_card["replay_pass_rate"] == pytest.approx(1.0)
    assert {action["id"] for action in replay_report_card["actions"]} == {
        "rerun_replay",
        "report_artifact",
    }
    replay_manifest_card = replay_report_card["manifests"][0]
    assert replay_manifest_card["status"] == "passed"
    assert replay_manifest_card["error_finding_count"] == 0
    assert replay_manifest_card["warning_finding_count"] == 4
    assert replay_manifest_card["metrics"]["framework_runtime_contract"] == pytest.approx(
        1.0
    )


def test_social_memory_framework_optimization_example_synthesizes_patches(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_SOCIAL_MEMORY_OPT_EXAMPLE_KEY",
        "real-local-social-memory-key",
    )

    output_path = tmp_path / "social-memory-framework-optimization.json"
    junit_path = tmp_path / "social-memory-framework-optimization.junit.xml"
    sarif_path = tmp_path / "social-memory-framework-optimization.sarif.json"
    markdown_path = tmp_path / "social-memory-framework-optimization.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "social_memory_framework_optimization.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.optimization.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] >= 0.95
    assert payload["summary"]["evaluation_score"] == pytest.approx(1.0)
    assert {"agent", "simulation.environments"} <= set(
        payload["summary"]["search_paths"]
    )

    best_agent = payload["optimization"]["best_config"]["agent"]
    assert best_agent["framework"] == "custom_refund_orchestrator"
    assert best_agent["method"] == "execute_task"
    assert best_agent["input_mode"] == "dict"
    best_env = payload["optimization"]["best_config"]["simulation"]["environments"][0]
    assert best_env["data"]["spans"][0]["signals"] == ["planner", "tool", "policy"]

    trace = payload["optimization"]["optimizer_trace"]
    assert trace["optimizer"] == "AgentSocialMemoryOptimizer"
    assert {role["name"] for role in trace["roles"]} >= {
        "seed",
        "smriti",
        "sangha",
        "dharma_steward",
    }
    best_proposal = next(
        proposal
        for proposal in trace["proposals"]
        if proposal["candidate_id"] == trace["best_candidate_id"]
    )
    assert best_proposal["role"] == "sangha"
    assert set(best_proposal["patch"]) == {"agent", "simulation.environments"}
    assert trace["summary"]["has_synthesis"] is True

    best_history = max(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["proposal_role"] == "sangha"
    assert best_history["metrics"]["framework_runtime_contract"] == pytest.approx(1.0)
    assert best_history["metrics"]["framework_runtime_coverage"] == pytest.approx(1.0)
    assert best_history["metrics"]["framework_trace_coverage"] == pytest.approx(1.0)

    case = best_history["report"]["results"][0]
    state = case["metadata"]["environment_state"]
    assert state["framework_runtime"]["summary"]["methods"] == ["execute_task"]
    assert state["framework_runtime"]["summary"]["input_modes"] == ["dict"]
    assert state["framework_runtime"]["summary"]["tool_call_count"] == 1
    assert state["framework_trace"]["adapter_conformance"]["passed"] is True

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "social-memory-framework-optimization" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_voice_streaming_realtime_manifest_runs_manifest_environments(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_VOICE_STREAMING_EXAMPLE_KEY",
        "real-local-voice-streaming-key",
    )

    output_path = tmp_path / "voice-streaming.json"
    junit_path = tmp_path / "voice-streaming.junit.xml"
    sarif_path = tmp_path / "voice-streaming.sarif.json"
    markdown_path = tmp_path / "voice-streaming.md"

    exit_code = main([
        "run",
        str(EXAMPLES / "voice_streaming_realtime_manifest.json"),
        "--no-eval",
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.run.v1"
    assert payload["status"] == "passed"
    case = payload["report"]["results"][0]
    state = case["metadata"]["environment_state"]
    assert set(state) >= {"voice", "streaming_trace"}

    voice = state["voice"]
    assert voice["sample_rate_hz"] == 16000
    assert voice["last_transcript"] == "I need help with a refund on my order."
    assert voice["route_history"] == [
        {
            "route": "support",
            "reason": "refund support request",
            "target": {"queue": "refund_support", "priority": "high"},
        }
    ]
    assert voice["timing_distribution"]["stage_order"] == ["vad", "stt", "llm", "tts"]
    assert voice["timing_distribution"]["sample_count"] == 12
    assert voice["timing_distribution"]["stages"]["tts"]["p50_ms"] == 260.0
    assert voice["tts_history"][0]["text"].startswith("Your refund request")

    streaming = state["streaming_trace"]
    assert streaming["framework"] == "livekit"
    assert streaming["summary"]["event_count"] == 4
    assert streaming["summary"]["tool_delta_count"] == 1
    assert "tool_delta" in streaming["signals"]

    assistant_tool_names = {
        call["name"]
        for message in case["messages"]
        if message["role"] == "assistant"
        for call in message.get("tool_calls", [])
    }
    assert {
        "voice_status",
        "voice_timing",
        "transcribe_audio",
        "route_call",
        "streaming_trace_status",
        "list_stream_events",
        "inspect_stream_event",
        "speak",
    } <= assistant_tool_names
    tool_response_ids = {
        message.get("tool_call_id")
        for message in case["messages"]
        if message["role"] == "tool"
    }
    assert {
        "voice_status",
        "voice_timing",
        "transcribe_user",
        "route_support",
        "stream_status",
        "stream_tool_events",
        "inspect_stream_tool",
        "speak_answer",
    } <= tool_response_ids

    event_names = {(event["type"], event.get("name")) for event in case["events"]}
    assert ("voice_trace", "voice_status") in event_names
    assert ("voice_timing", "voice_timing_distribution") in event_names
    assert ("voice_route", "call_routed") in event_names
    assert ("voice", "tts_output") in event_names
    assert ("streaming_trace", "streaming_trace_status") in event_names
    assert ("streaming_trace", "streaming_events_listed") in event_names
    assert ("streaming_trace", "streaming_event_inspected") in event_names

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "voice-streaming-realtime-simulation" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_voice_streaming_realtime_optimization_example_runs_evidence_gates(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_VOICE_STREAMING_OPT_EXAMPLE_KEY",
        "real-local-voice-streaming-opt-key",
    )

    output_path = tmp_path / "voice-streaming-optimization.json"
    junit_path = tmp_path / "voice-streaming-optimization.junit.xml"
    sarif_path = tmp_path / "voice-streaming-optimization.sarif.json"
    markdown_path = tmp_path / "voice-streaming-optimization.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "voice_streaming_realtime_optimization.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.optimization.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] >= 0.99
    assert payload["summary"]["evaluation_score"] == pytest.approx(1.0)
    assert "simulation.environments" in payload["summary"]["search_paths"]

    best_config = payload["optimization"]["best_config"]
    env_types = [
        environment["type"]
        for environment in best_config["simulation"]["environments"]
    ]
    assert env_types == ["voice", "streaming_trace"]
    assert best_config["simulation"]["environments"][0]["data"]["sample_rate_hz"] == 16000
    assert (
        best_config["simulation"]["environments"][1]["data"]["state"]["route"]
        == "support"
    )

    best_history = max(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["patch"].keys() == {"simulation.environments"}
    metrics = best_history["metrics"]
    for metric in (
        "voice_trace_coverage",
        "voice_interaction_quality",
        "voice_timing_distribution_quality",
        "voice_turn_taking",
        "tool_argument_schema",
        "streaming_trace_coverage",
        "streaming_interaction_quality",
    ):
        assert metrics[metric] == pytest.approx(1.0)

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "voice-streaming-realtime-optimization" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_redteam_campaign_optimization_example_runs_evidence_gates(
    tmp_path,
    monkeypatch,
):
    from fi.alk import optimize

    monkeypatch.setenv(
        "AGENT_LEARNING_REDTEAM_OPT_EXAMPLE_KEY",
        "real-local-redteam-opt-key",
    )

    output_path = tmp_path / "redteam-campaign-optimization.json"
    junit_path = tmp_path / "redteam-campaign-optimization.junit.xml"
    sarif_path = tmp_path / "redteam-campaign-optimization.sarif.json"
    markdown_path = tmp_path / "redteam-campaign-optimization.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "redteam_campaign_optimization.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.optimization.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] >= 0.9
    assert payload["summary"]["evaluation_score"] == pytest.approx(1.0)
    assert "simulation.environments" in payload["summary"]["search_paths"]

    manifest = json.loads(
        (EXAMPLES / "redteam_campaign_optimization.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["optimization"]["scoring"]["method"] == "simulation_evidence"
    assert manifest["optimization"]["scoring"]["layers"] == [
        "red_team_readiness"
    ]
    assert {item["year"] for item in manifest["optimization"]["target"]["metadata"]["research_sources"]} == {2026}

    best_config = payload["optimization"]["best_config"]
    env_types = [
        environment["type"]
        for environment in best_config["simulation"]["environments"]
    ]
    assert env_types == [
        "adversarial_attack_pack",
        "red_team_campaign",
        "red_team_readiness",
    ]

    best_campaign = best_config["simulation"]["environments"][1]["data"]
    assert best_campaign["required_attack_types"] == [
        "prompt_injection",
        "credential_exfiltration",
    ]
    assert best_campaign["required_surfaces"] == ["tool", "memory"]

    best_history = max(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["patch"].keys() == {"simulation.environments"}
    metrics = best_history["metrics"]
    for metric in (
        "adversarial_resilience",
        "red_team_campaign_coverage",
        "red_team_campaign_quality",
        "red_team_readiness_coverage",
        "red_team_readiness_quality",
        "tool_selection_accuracy",
    ):
        assert metrics[metric] == pytest.approx(1.0)

    state = best_history["report"]["results"][0]["metadata"]["environment_state"]
    readiness_summary = state["red_team_readiness"]["summary"]
    assert readiness_summary["ready_components"] == [
        "control_plane",
        "framework_import",
        "red_team_campaign",
        "trust_boundary",
        "workspace_run",
    ]
    assert readiness_summary["blocking_gaps"] == []
    assert readiness_summary["blocking_gap_count"] == 0

    candidate = optimize.AgentCandidate.from_config(
        payload["optimization"]["best_config"],
        layers=manifest["optimization"]["target"]["layers"],
    )
    evidence = optimize.score_simulation_evidence(
        best_history["report"],
        manifest=manifest,
        candidate=candidate,
        config=manifest["optimization"]["scoring"],
    )
    assert evidence.score == pytest.approx(1.0)
    assert {
        item["name"]: item["score"]
        for item in evidence.metadata["simulation_evidence_score"]["components"]
    } == {
        "tool_coverage": 1.0,
        "red_team_readiness": 1.0,
    }

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "redteam-campaign-optimization" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_redteam_autogen_optimization_example_regenerates_candidate_matrix(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_REDTEAM_AUTOGEN_OPT_EXAMPLE_KEY",
        "real-local-redteam-autogen-opt-key",
    )

    output_path = tmp_path / "redteam-autogen-optimization.json"
    junit_path = tmp_path / "redteam-autogen-optimization.junit.xml"
    sarif_path = tmp_path / "redteam-autogen-optimization.sarif.json"
    markdown_path = tmp_path / "redteam-autogen-optimization.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "redteam_autogen_optimization.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.optimization.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] >= 0.97
    assert payload["summary"]["evaluation_score"] == pytest.approx(1.0)
    assert payload["redteam"]["auto_generate"] is True
    assert set(payload["summary"]["search_paths"]) >= {
        "redteam.attacks",
        "redteam.surfaces",
    }

    best_config = payload["optimization"]["best_config"]
    assert best_config["redteam"]["attacks"] == [
        "prompt_injection",
        "credential_exfiltration",
    ]
    assert best_config["redteam"]["surfaces"] == ["tool", "memory"]

    best_history = max(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["patch"] == {
        "redteam.attacks": [
            "prompt_injection",
            "credential_exfiltration",
        ],
        "redteam.surfaces": ["tool", "memory"],
    }
    metrics = best_history["metrics"]
    for metric in (
        "adversarial_resilience",
        "red_team_campaign_coverage",
        "red_team_campaign_quality",
        "tool_selection_accuracy",
    ):
        assert metrics[metric] == pytest.approx(1.0)

    case = best_history["report"]["results"][0]
    state = case["metadata"]["environment_state"]
    assert set(state) >= {"adversarial", "red_team_campaign"}
    campaign_summary = state["red_team_campaign"]["summary"]
    assert campaign_summary["attack_count"] == 4
    assert campaign_summary["coverage_cell_count"] == 4
    assert campaign_summary["missing_coverage_cells"] == []
    assert campaign_summary["missing_executed_cells"] == []

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "redteam-autogen-optimization" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_long_horizon_redteam_optimization_example_selects_attack_system(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_LONG_HORIZON_REDTEAM_OPT_EXAMPLE_KEY",
        "real-local-long-horizon-redteam-opt-key",
    )

    output_path = tmp_path / "long-horizon-redteam-optimization.json"
    junit_path = tmp_path / "long-horizon-redteam-optimization.junit.xml"
    sarif_path = tmp_path / "long-horizon-redteam-optimization.sarif.json"
    markdown_path = tmp_path / "long-horizon-redteam-optimization.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "long_horizon_redteam_optimization.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    attacks = [
        "intent_hijacking",
        "task_injection",
        "objective_drift",
        "tool_chaining",
        "memory_poisoning",
    ]
    surfaces = ["instruction", "tool", "memory", "retrieval", "environment"]

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.optimization.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] >= 0.95
    assert payload["summary"]["evaluation_score"] == pytest.approx(1.0)
    assert "redteam" in payload["summary"]["search_paths"]

    best_redteam = payload["optimization"]["best_config"]["redteam"]
    assert best_redteam["profile"] == "stateful_attack_system"
    assert best_redteam["preset"] == "long_horizon_agent"
    assert best_redteam["attacks"] == attacks
    assert best_redteam["surfaces"] == surfaces
    assert best_redteam["signals"] == [
        "research_backed",
        "long_horizon",
        "stateful",
        "multi_turn",
        "objective_integrity",
        "tool_chain",
        "memory_poisoning",
        "compositional_orchestration",
        "pre_deployment_telemetry",
        "persistent_memory",
    ]
    assert best_redteam["attack_system"]["strategy"] == (
        "long_horizon_stateful_campaign"
    )
    assert best_redteam["attack_system"]["planner"] == "campaign_matrix"
    assert {source["source"] for source in best_redteam["attack_system"]["research_basis"]} >= {
        "arxiv:2601.13518",
        "arxiv:2602.16346",
        "arxiv:2605.01970",
    }

    best_history = max(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["patch"].keys() == {"redteam"}
    metrics = best_history["metrics"]
    for metric in (
        "adversarial_resilience",
        "red_team_campaign_coverage",
        "red_team_campaign_quality",
        "tool_selection_accuracy",
    ):
        assert metrics[metric] == pytest.approx(1.0)

    state = best_history["report"]["results"][0]["metadata"]["environment_state"]
    campaign_summary = state["red_team_campaign"]["summary"]
    assert campaign_summary["attack_count"] == 25
    assert campaign_summary["coverage_cell_count"] == 25
    assert campaign_summary["executed_cell_count"] == 25
    assert campaign_summary["multi_turn_scenario_count"] == 25
    assert campaign_summary["missing_coverage_cells"] == []
    assert campaign_summary["missing_executed_cells"] == []

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "long-horizon-redteam-optimization" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_persistent_state_redteam_optimization_example_selects_hardened_policy(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_PERSISTENT_REDTEAM_OPT_EXAMPLE_KEY",
        "real-local-persistent-redteam-opt-key",
    )

    output_path = tmp_path / "persistent-state-redteam-optimization.json"
    junit_path = tmp_path / "persistent-state-redteam-optimization.junit.xml"
    sarif_path = tmp_path / "persistent-state-redteam-optimization.sarif.json"
    markdown_path = tmp_path / "persistent-state-redteam-optimization.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "persistent_state_redteam_optimization.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.optimization.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] >= 0.99
    assert payload["summary"]["evaluation_score"] == pytest.approx(1.0)
    assert "simulation.environments" in payload["summary"]["search_paths"]

    manifest = json.loads(
        (EXAMPLES / "persistent_state_redteam_optimization.json").read_text(
            encoding="utf-8"
        )
    )
    assert {
        item["year"]
        for item in manifest["optimization"]["target"]["metadata"]["research_sources"]
    } == {2026}

    best_env = payload["optimization"]["best_config"]["simulation"][
        "environments"
    ][0]
    assert best_env["type"] == "persistent_state_attack"
    assert best_env["data"]["metadata"]["profile"] == "hardened"
    assert best_env["data"]["metadata"]["controls"] == {
        "write_policy": "quarantine_untrusted_durable_writes",
        "context_rehydration": "trusted_context_only",
        "activation_guard": "block_untrusted_persistent_instruction",
    }

    best_history = max(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["patch"].keys() == {"simulation.environments"}
    metrics = best_history["metrics"]
    assert metrics["persistent_state_attack_coverage"] == pytest.approx(1.0)
    assert metrics["persistent_state_attack_quality"] == pytest.approx(1.0)
    state = best_history["report"]["results"][0]["metadata"]["environment_state"]
    summary = state["persistent_state_attack"]["summary"]
    assert summary["case_count"] == 1
    assert summary["write_attempt_count"] == 1
    assert summary["incorporation_attempt_count"] == 1
    assert summary["activation_attempt_count"] == 1
    assert summary["write_success_rate"] == 0.0
    assert summary["incorporation_rate"] == 0.0
    assert summary["activation_rate"] == 0.0
    assert summary["e2e_attack_success_rate"] == 0.0

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "persistent-state-redteam-optimization" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_persistent_state_redteam_optimization_promotes_replayable_regression(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_PERSISTENT_REDTEAM_OPT_EXAMPLE_KEY",
        "real-local-persistent-redteam-opt-key",
    )

    output_path = tmp_path / "persistent-state-redteam-optimization.json"
    promotion_path = tmp_path / "persistent-state-redteam-promotion.json"
    manifest_path = tmp_path / "persistent-state-redteam-regression.json"
    replay_path = tmp_path / "persistent-state-redteam-replay.json"
    junit_path = tmp_path / "persistent-state-redteam-replay.junit.xml"
    sarif_path = tmp_path / "persistent-state-redteam-replay.sarif.json"
    markdown_path = tmp_path / "persistent-state-redteam-replay.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "persistent_state_redteam_optimization.json"),
        "--output",
        str(output_path),
    ])
    assert exit_code == 0

    exit_code = main([
        "promote-to-regression",
        str(output_path),
        "--output",
        str(promotion_path),
        "--manifest",
        str(manifest_path),
        "--min-level",
        "note",
        "--max-findings",
        "1",
        "--required-env",
        "AGENT_LEARNING_PERSISTENT_REDTEAM_REGRESSION_KEY",
    ])
    assert exit_code == 0

    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    assert promotion["status"] == "passed"
    assert promotion["summary"]["promotion_kind"] == "persistent_state_optimization"
    assert promotion["summary"]["promoted_finding_count"] == 0
    assert promotion["summary"]["promoted_manifest_count"] == 1
    assert promotion["summary"]["best_profile"] == "hardened"
    assert promotion["summary"]["write_success_rate"] == pytest.approx(0.0)
    assert promotion["summary"]["incorporation_rate"] == pytest.approx(0.0)
    assert promotion["summary"]["activation_rate"] == pytest.approx(0.0)
    assert promotion["summary"]["e2e_attack_success_rate"] == pytest.approx(0.0)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["required_env"] == [
        "AGENT_LEARNING_PERSISTENT_REDTEAM_REGRESSION_KEY"
    ]
    env = manifest["simulation"]["environments"][0]
    assert env["type"] == "persistent_state_attack"
    assert env["data"]["metadata"]["profile"] == "hardened"
    config = manifest["evaluation"]["agent_report"]["config"]
    assert config["required_tools"] == [
        "persistent_state_attack_status",
        "list_persistent_state_writes",
        "list_persistent_state_incorporations",
        "list_persistent_state_activations",
        "list_persistent_state_gaps",
    ]
    assert {
        "persistent_state_attack",
        "stored_prompt_injection",
        "cross_session",
        "session_reset",
        "write_stage",
        "incorporation_stage",
        "activation_stage",
        "provenance",
        "memory",
    } <= set(config["required_persistent_state_attack"])
    quality = config["persistent_state_attack_quality"]
    assert quality["min_case_count"] == 1
    assert quality["max_written_count"] == 0
    assert quality["max_incorporated_count"] == 0
    assert quality["max_activated_count"] == 0
    assert quality["max_e2e_attack_success_rate"] == 0.0
    assert quality["require_session_reset"] is True
    assert quality["require_no_missing_provenance"] is True

    monkeypatch.setenv(
        "AGENT_LEARNING_PERSISTENT_REDTEAM_REGRESSION_KEY",
        "real-local-persistent-redteam-regression-key",
    )
    exit_code = main([
        "replay",
        str(manifest_path),
        "--output",
        str(replay_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])
    assert exit_code == 0

    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    assert replay["status"] == "passed"
    assert replay["summary"]["replay_pass_rate"] == pytest.approx(1.0)
    child = replay["replay"]["manifests"][0]
    assert child["command"] == "run"
    metrics = child["summary"]["metric_averages"]
    assert metrics["persistent_state_attack_coverage"] == pytest.approx(1.0)
    assert metrics["persistent_state_attack_quality"] == pytest.approx(1.0)

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "persistent-state-redteam-regression" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_redteam_society_optimization_example_selects_council(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_REDTEAM_SOCIETY_OPT_EXAMPLE_KEY",
        "real-local-redteam-society-opt-key",
    )

    output_path = tmp_path / "redteam-society-optimization.json"
    junit_path = tmp_path / "redteam-society-optimization.junit.xml"
    sarif_path = tmp_path / "redteam-society-optimization.sarif.json"
    markdown_path = tmp_path / "redteam-society-optimization.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "redteam_society_optimization.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    roles = {
        "red_team_lead",
        "orchestrator_leak_tester",
        "tool_chain_attacker",
        "memory_privacy_guard",
        "vidura",
        "dharma_steward",
    }

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.optimization.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] >= 0.96
    assert payload["summary"]["evaluation_score"] == pytest.approx(1.0)
    assert "simulation.environments" in payload["summary"]["search_paths"]

    best_history = max(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["patch"].keys() == {"simulation.environments"}
    metrics = best_history["metrics"]
    for metric in (
        "adversarial_resilience",
        "red_team_campaign_coverage",
        "red_team_campaign_quality",
        "multi_agent_trace_coverage",
        "multi_agent_coordination_quality",
        "tool_selection_accuracy",
    ):
        assert metrics[metric] == pytest.approx(1.0)

    best_room = payload["optimization"]["best_config"]["simulation"][
        "environments"
    ][0]["data"]
    assert set(best_room["participants"]) == roles
    assert best_room["allow_unknown_roles"] is False
    assert len(best_room["expected_handoffs"]) == 3
    assert best_room["expected_reconciliation"]["accepted_source"] == (
        "dharma_steward"
    )

    state = best_history["report"]["results"][0]["metadata"]["environment_state"]
    assert set(state) == {"adversarial", "multi_agent", "red_team_campaign"}
    multi_agent = state["multi_agent"]
    assert set(multi_agent["participants"]) == roles
    assert len(multi_agent["handoffs"]) == 3
    assert len(multi_agent["reviews"]) == 1
    assert len(multi_agent["reconciliations"]) == 1
    assert all(check["match"] for check in multi_agent["coordination_checks"])
    campaign_summary = state["red_team_campaign"]["summary"]
    assert campaign_summary["attack_count"] == 25
    assert campaign_summary["coverage_cell_count"] == 25
    assert campaign_summary["executed_cell_count"] == 25
    assert campaign_summary["missing_coverage_cells"] == []
    assert campaign_summary["missing_executed_cells"] == []

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "redteam-society-optimization" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_redteam_causal_attribution_optimization_example_selects_graph(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_REDTEAM_CAUSAL_ATTRIBUTION_OPT_EXAMPLE_KEY",
        "real-local-redteam-causal-opt-key",
    )

    output_path = tmp_path / "redteam-causal-attribution-optimization.json"
    junit_path = tmp_path / "redteam-causal-attribution-optimization.junit.xml"
    sarif_path = tmp_path / "redteam-causal-attribution-optimization.sarif.json"
    markdown_path = tmp_path / "redteam-causal-attribution-optimization.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "redteam_causal_attribution_optimization.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    roles = {
        "red_team_lead",
        "orchestrator_leak_tester",
        "tool_chain_attacker",
        "memory_privacy_guard",
        "vidura",
        "dharma_steward",
    }
    required_nodes = {
        "user_prompt",
        "orchestrator",
        "retriever",
        "memory_store",
        "tool_executor",
        "critic",
        "dharma_steward",
    }
    required_root_causes = {
        "orchestrator_delegation_boundary",
        "memory_persistence_without_quarantine",
        "tool_chain_without_approval_gate",
    }

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.optimization.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] >= 0.96
    assert payload["summary"]["evaluation_score"] == pytest.approx(1.0)
    assert "simulation.environments" in payload["summary"]["search_paths"]

    best_history = max(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["patch"].keys() == {"simulation.environments"}
    metrics = best_history["metrics"]
    for metric in (
        "adversarial_resilience",
        "red_team_campaign_coverage",
        "red_team_campaign_quality",
        "multi_agent_trace_coverage",
        "multi_agent_coordination_quality",
        "causal_attribution_quality",
        "tool_selection_accuracy",
    ):
        assert metrics[metric] == pytest.approx(1.0)

    best_room = payload["optimization"]["best_config"]["simulation"][
        "environments"
    ][0]["data"]
    assert set(best_room["participants"]) == roles
    assert best_room["allow_unknown_roles"] is False
    causal_graph = best_room["state"]["causal_attribution"]
    assert {node["id"] for node in causal_graph["nodes"]} == required_nodes
    assert len(causal_graph["edges"]) == 7
    assert {item["id"] for item in causal_graph["root_causes"]} == (
        required_root_causes
    )
    assert len(causal_graph["mitigations"]) == 4
    assert len(causal_graph["evidence"]) == 5

    state = best_history["report"]["results"][0]["metadata"]["environment_state"]
    assert set(state) == {"adversarial", "multi_agent", "red_team_campaign"}
    multi_agent = state["multi_agent"]
    assert set(multi_agent["participants"]) == roles
    assert all(check["match"] for check in multi_agent["coordination_checks"])
    observed_graph = multi_agent["state"]["causal_attribution"]
    assert {node["id"] for node in observed_graph["nodes"]} == required_nodes

    agent_report = best_history["report"]["results"][0]["evaluation"]["agent_report"]
    causal_metric = next(
        item for item in agent_report["metrics"]
        if item["name"] == "causal_attribution_quality"
    )
    observed = causal_metric["details"]["observed"]
    assert causal_metric["score"] == pytest.approx(1.0)
    assert set(observed["nodes"]) == required_nodes
    assert set(observed["root_causes"]) == required_root_causes
    assert observed["mapped_root_causes"] == sorted(required_root_causes)
    assert observed["unmapped_root_causes"] == []
    assert observed["is_dag"] is True
    campaign_summary = state["red_team_campaign"]["summary"]
    assert campaign_summary["attack_count"] == 25
    assert campaign_summary["coverage_cell_count"] == 25
    assert campaign_summary["executed_cell_count"] == 25

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "redteam-causal-attribution-optimization" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_report_repair_optimization_example_scores_simulation_evidence(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_REPORT_REPAIR_OPT_EXAMPLE_KEY",
        "real-local-report-repair-opt-key",
    )

    output_path = tmp_path / "report-repair-optimization.json"
    junit_path = tmp_path / "report-repair-optimization.junit.xml"
    sarif_path = tmp_path / "report-repair-optimization.sarif.json"
    markdown_path = tmp_path / "report-repair-optimization.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "report_repair_optimization.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.optimization.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] == pytest.approx(1.0)
    assert payload["summary"]["evaluation_passed"] is True
    assert "simulation.environments" in payload["summary"]["search_paths"]

    best_history = max(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["patch"].keys() == {"simulation.environments"}
    assert best_history["score"] == pytest.approx(1.0)
    metrics = best_history["metrics"]
    for metric in (
        "tool_selection_accuracy",
        "framework_trace_coverage",
        "agent_memory_lineage_quality",
        "orchestration_flow_quality",
        "world_contract_quality",
    ):
        assert metrics[metric] == pytest.approx(1.0)

    state = best_history["report"]["results"][0]["metadata"]["environment_state"]
    assert set(state) == {
        "adversarial",
        "agent_memory_lineage",
        "framework_trace",
        "orchestration_trace",
        "world_attack_replay",
        "world_contract",
        "world_orchestration_replay",
    }
    assert state["world_contract"]["summary"]["terminal_status"] == "success"
    assert state["world_contract"]["summary"][
        "completed_required_transition_count"
    ] == 1
    assert state["agent_memory_lineage"]["summary"]["has_audit"] is True
    assert state["framework_trace"]["adapter_conformance"]["passed"] is True

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "report-repair-optimization" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_framework_import_repair_optimization_example_scores_import_evidence(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_FRAMEWORK_IMPORT_REPAIR_OPT_EXAMPLE_KEY",
        "real-local-framework-import-repair-opt-key",
    )

    output_path = tmp_path / "framework-import-repair-optimization.json"
    junit_path = tmp_path / "framework-import-repair-optimization.junit.xml"
    sarif_path = tmp_path / "framework-import-repair-optimization.sarif.json"
    markdown_path = tmp_path / "framework-import-repair-optimization.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "framework_import_repair_optimization.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.optimization.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] == pytest.approx(1.0)
    assert payload["summary"]["evaluation_passed"] is True
    assert "simulation.environments" in payload["summary"]["search_paths"]

    best_history = max(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["patch"].keys() == {"simulation.environments"}
    assert best_history["score"] == pytest.approx(1.0)
    metrics = best_history["metrics"]
    for metric in (
        "tool_selection_accuracy",
        "framework_import_coverage",
        "framework_import_quality",
    ):
        assert metrics[metric] == pytest.approx(1.0)

    state = best_history["report"]["results"][0]["metadata"]["environment_state"]
    assert set(state) == {"framework_import_manifest"}
    summary = state["framework_import_manifest"]["summary"]
    assert summary["source_count"] == 24
    assert summary["passed_source_count"] == 24
    assert summary["failed_source_count"] == 0
    assert summary["missing_required_frameworks"] == []
    assert summary["missing_required_export_types"] == []
    assert summary["missing_required_signals"] == []
    assert summary["observed_frameworks"] == [
        "langchain",
        "langgraph",
        "livekit",
        "pipecat",
    ]
    assert summary["observed_export_types"] == [
        "capability_matrix",
        "event_stream",
        "lifecycle",
        "portability_matrix",
        "probe_suite",
        "trace_export",
    ]

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "framework-import-repair-optimization" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_workspace_observability_optimization_example_runs_evidence_gates(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_WORKSPACE_OBSERVABILITY_OPT_EXAMPLE_KEY",
        "real-local-workspace-observability-opt-key",
    )

    output_path = tmp_path / "workspace-observability-optimization.json"
    junit_path = tmp_path / "workspace-observability-optimization.junit.xml"
    sarif_path = tmp_path / "workspace-observability-optimization.sarif.json"
    markdown_path = tmp_path / "workspace-observability-optimization.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "workspace_observability_optimization.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.optimization.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] >= 0.9
    assert payload["summary"]["evaluation_score"] == pytest.approx(1.0)
    assert "simulation.environments" in payload["summary"]["search_paths"]

    env_types = [
        environment["type"]
        for environment in payload["optimization"]["best_config"]["simulation"][
            "environments"
        ]
    ]
    assert env_types == ["workspace_run_manifest", "observability_replay"]

    best_history = max(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["patch"].keys() == {"simulation.environments"}
    metrics = best_history["metrics"]
    for metric in (
        "workspace_run_coverage",
        "workspace_run_quality",
        "observability_replay_coverage",
        "observability_replay_quality",
        "tool_selection_accuracy",
    ):
        assert metrics[metric] == pytest.approx(1.0)

    case = best_history["report"]["results"][0]
    state = case["metadata"]["environment_state"]
    assert set(state) >= {"workspace_run_manifest", "observability_replay_pack"}
    workspace_summary = state["workspace_run_manifest"]["summary"]
    assert workspace_summary["failed_command_count"] == 0
    assert workspace_summary["open_red_team_finding_count"] == 0
    assert workspace_summary["secret_leak_count"] == 0
    assert workspace_summary["missing_required_evidence"] == []
    replay_summary = state["observability_replay_pack"]["summary"]
    assert replay_summary["case_count"] == 2
    assert replay_summary["failed_case_count"] == 1
    assert replay_summary["missing_trace_signals"] == []

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "workspace-observability-optimization" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_agent_integration_optimization_example_runs_provider_matrix(
    tmp_path,
    monkeypatch,
):
    from fi.alk import optimize

    monkeypatch.setenv(
        "AGENT_LEARNING_AGENT_INTEGRATION_OPT_EXAMPLE_KEY",
        "real-local-agent-integration-opt-key",
    )

    output_path = tmp_path / "agent-integration-optimization.json"
    junit_path = tmp_path / "agent-integration-optimization.junit.xml"
    sarif_path = tmp_path / "agent-integration-optimization.sarif.json"
    markdown_path = tmp_path / "agent-integration-optimization.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "agent_integration_optimization.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.optimization.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] >= 0.98
    assert payload["summary"]["evaluation_score"] == pytest.approx(1.0)
    assert "simulation.environments" in payload["summary"]["search_paths"]
    manifest = json.loads(
        (EXAMPLES / "agent_integration_optimization.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["optimization"]["scoring"]["method"] == "simulation_evidence"
    assert manifest["optimization"]["scoring"]["layers"] == ["agent_integration"]
    assert {
        item["year"]
        for item in manifest["optimization"]["target"]["metadata"]["research_sources"]
    } == {2026}

    env_types = [
        environment["type"]
        for environment in payload["optimization"]["best_config"]["simulation"][
            "environments"
        ]
    ]
    assert env_types == ["agent_integration"]

    best_history = max(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["patch"].keys() == {"simulation.environments"}
    metrics = best_history["metrics"]
    for metric in (
        "agent_integration_coverage",
        "agent_integration_quality",
        "tool_selection_accuracy",
    ):
        assert metrics[metric] == pytest.approx(1.0)

    case = best_history["report"]["results"][0]
    quality_metric = next(
        metric
        for metric in case["evaluation"]["agent_report"]["metrics"]
        if metric["name"] == "agent_integration_quality"
    )
    provider_channel_checks = {
        (check["expected"]["provider"], check["expected"]["channel"]): check["match"]
        for check in quality_metric["details"]["checks"]
        if check["check"] == "required_provider_channel"
    }
    assert provider_channel_checks[("vapi", "phone")] is True
    assert provider_channel_checks[("vapi", "webrtc")] is True
    assert provider_channel_checks[("bland", "phone")] is True
    assert provider_channel_checks[("bland", "sip")] is True
    assert provider_channel_checks[("bland", "webrtc")] is True

    state = case["metadata"]["environment_state"]
    assert set(state) == {"agent_integration_manifest"}
    summary = state["agent_integration_manifest"]["summary"]
    assert set(summary["observed_providers"]) >= {
        "agora",
        "bland",
        "deepgram",
        "elevenlabs",
        "livekit",
        "pipecat",
        "retell",
        "twilio",
        "vapi",
    }
    assert set(summary["observed_channels"]) >= {
        "chat",
        "voice",
        "webrtc",
        "phone",
        "sip",
        "websocket",
        "media_stream",
    }
    assert set(summary["trace_frameworks"]) >= {
        "autogen",
        "crewai",
        "langchain",
        "langgraph",
        "livekit",
        "openai_agents",
        "pipecat",
    }
    assert summary["verified_provider_count"] == 16
    assert summary["failed_session_count"] == 0
    assert summary["missing_required_providers"] == []
    assert summary["missing_required_channels"] == []
    assert summary["missing_required_trace_frameworks"] == []
    assert summary["providers_without_verified_credentials"] == []

    candidate = optimize.AgentCandidate.from_config(
        payload["optimization"]["best_config"],
        layers=manifest["optimization"]["target"]["layers"],
    )
    evidence = optimize.score_simulation_evidence(
        best_history["report"],
        manifest=manifest,
        candidate=candidate,
        config=manifest["optimization"]["scoring"],
    )
    assert evidence.score == pytest.approx(1.0)
    assert {
        item["name"]: item["score"]
        for item in evidence.metadata["simulation_evidence_score"]["components"]
    } == {
        "tool_coverage": 1.0,
        "agent_integration": 1.0,
    }

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "agent-integration-optimization" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_multi_agent_framework_handoff_optimization_example_runs_captured_traces(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_MULTI_AGENT_FRAMEWORK_HANDOFF_OPT_EXAMPLE_KEY",
        "real-local-multi-agent-framework-handoff-opt-key",
    )

    output_path = tmp_path / "multi-agent-framework-handoff-optimization.json"
    junit_path = tmp_path / "multi-agent-framework-handoff-optimization.junit.xml"
    sarif_path = tmp_path / "multi-agent-framework-handoff-optimization.sarif.json"
    markdown_path = tmp_path / "multi-agent-framework-handoff-optimization.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "multi_agent_framework_handoff_optimization.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.optimization.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] >= 0.99
    assert payload["summary"]["evaluation_score"] == pytest.approx(1.0)
    assert payload["optimization"]["optimizer_trace"]["optimizer"] == (
        "AgentEvolutionOptimizer"
    )
    assert "simulation.environments" in payload["summary"]["search_paths"]

    best_config_envs = payload["optimization"]["best_config"]["simulation"][
        "environments"
    ]
    assert [environment["type"] for environment in best_config_envs] == [
        "framework_trace",
        "framework_trace",
        "framework_trace",
        "framework_trace",
        "multi_agent_room",
    ]
    assert [
        environment["data"]["framework"]
        for environment in best_config_envs
        if environment["type"] == "framework_trace"
    ] == ["openai_agents", "autogen", "crewai", "langgraph"]

    best_history = max(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["patch"].keys() == {"simulation.environments"}
    metrics = best_history["metrics"]
    for metric in (
        "framework_transcript_quality",
        "multi_agent_trace_coverage",
        "multi_agent_coordination_quality",
        "task_completion",
        "trajectory_score",
    ):
        assert metrics[metric] == pytest.approx(1.0)

    case = best_history["report"]["results"][0]
    state = case["metadata"]["environment_state"]
    assert set(state) == {"framework_trace", "multi_agent"}

    transcript_metric = next(
        metric
        for metric in case["evaluation"]["agent_report"]["metrics"]
        if metric["name"] == "framework_transcript_quality"
    )
    observed = transcript_metric["details"]["observed"]
    assert set(observed["speaker_sequence"]) >= {
        "triage_agent",
        "retrieval_agent",
        "critic_agent",
        "planner",
        "researcher",
        "reviewer",
        "manager",
        "analyst",
        "qa",
        "retriever",
        "critic",
    }
    assert {handoff["to"] for handoff in observed["handoffs"]} >= {
        "retrieval_agent",
        "critic_agent",
        "researcher",
        "analyst",
        "retriever",
    }
    assert "ckpt_retrieval" in {
        checkpoint["id"].replace("-", "_")
        for checkpoint in observed["checkpoints"]
    }
    assert observed["errors"] == []

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "multi-agent-framework-handoff-optimization" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_optimizer_governance_optimization_example_runs_society_trace(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_OPTIMIZER_GOVERNANCE_OPT_EXAMPLE_KEY",
        "real-local-optimizer-governance-opt-key",
    )

    output_path = tmp_path / "optimizer-governance-optimization.json"
    junit_path = tmp_path / "optimizer-governance-optimization.junit.xml"
    sarif_path = tmp_path / "optimizer-governance-optimization.sarif.json"
    markdown_path = tmp_path / "optimizer-governance-optimization.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "optimizer_governance_optimization.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.optimization.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] >= 0.98
    assert payload["summary"]["evaluation_score"] == pytest.approx(1.0)
    assert "simulation.environments" in payload["summary"]["search_paths"]

    env_types = [
        environment["type"]
        for environment in payload["optimization"]["best_config"]["simulation"][
            "environments"
        ]
    ]
    assert env_types == ["optimizer_trace"]

    best_history = max(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["patch"].keys() == {"simulation.environments"}
    metrics = best_history["metrics"]
    for metric in (
        "optimizer_trace_coverage",
        "optimizer_trace_quality",
        "tool_selection_accuracy",
    ):
        assert metrics[metric] == pytest.approx(1.0)

    case = best_history["report"]["results"][0]
    state = case["metadata"]["environment_state"]
    assert set(state) == {"optimizer_society_trace"}
    trace_summary = state["optimizer_society_trace"]["summary"]
    assert trace_summary["role_count"] == 5
    assert trace_summary["proposal_count"] == 5
    assert trace_summary["round_count"] == 3
    assert trace_summary["diagnostic_count"] == 2
    assert trace_summary["role_credit_count"] == 5
    assert trace_summary["duplicate_candidate_count"] == 0
    assert trace_summary["best_candidate_id"] == "c_steward"
    assert trace_summary["final_score"] == pytest.approx(0.99)
    for flag in (
        "has_role_graph",
        "has_critique",
        "has_synthesis",
        "has_steward",
        "has_governance",
        "has_role_diversity",
        "has_mediator",
        "has_contract_gate",
        "has_rollback",
        "has_locality",
        "has_dependency_audit",
    ):
        assert trace_summary[flag] is True
    for flag in (
        "has_guna_axes",
        "has_two_chamber",
        "has_nyaya_justifications",
        "has_hetvabhasa_rejections",
        "has_nirnaya",
        "has_staged_conditioning",
        "has_layer_locality",
        "has_declared_budget",
        "has_external_ranking",
    ):
        assert trace_summary[flag] is True
    # Phase 4: the governed twin is engine-built — 11 computed checks plus
    # the 6 conditional society checks (explicit checks dedupe in).
    assert trace_summary["governance_check_count"] == 17
    assert trace_summary["governance_pass_rate"] == pytest.approx(1.0)

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "optimizer-governance-optimization" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_agent_control_plane_optimization_example_runs_trust_and_control_gate(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_AGENT_CONTROL_PLANE_OPT_EXAMPLE_KEY",
        "real-local-agent-control-plane-opt-key",
    )

    output_path = tmp_path / "agent-control-plane-optimization.json"
    junit_path = tmp_path / "agent-control-plane-optimization.junit.xml"
    sarif_path = tmp_path / "agent-control-plane-optimization.sarif.json"
    markdown_path = tmp_path / "agent-control-plane-optimization.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "agent_control_plane_optimization.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.optimization.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] >= 0.98
    assert payload["summary"]["evaluation_score"] == pytest.approx(1.0)
    assert "simulation.environments" in payload["summary"]["search_paths"]

    env_types = [
        environment["type"]
        for environment in payload["optimization"]["best_config"]["simulation"][
            "environments"
        ]
    ]
    assert env_types == ["agent_trust_boundary", "agent_control_plane"]

    best_history = max(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["patch"].keys() == {"simulation.environments"}
    metrics = best_history["metrics"]
    for metric in (
        "agent_trust_boundary_coverage",
        "agent_trust_boundary_quality",
        "agent_control_plane_coverage",
        "agent_control_plane_quality",
        "tool_selection_accuracy",
    ):
        assert metrics[metric] == pytest.approx(1.0)

    case = best_history["report"]["results"][0]
    state = case["metadata"]["environment_state"]
    assert set(state) == {"agent_trust_boundary_model", "agent_control_plane"}
    trust_summary = state["agent_trust_boundary_model"]["summary"]
    assert trust_summary["control_count"] == 11
    assert trust_summary["required_control_rate"] == pytest.approx(1.0)
    assert trust_summary["high_risk_unmitigated_count"] == 0
    assert trust_summary["gaps"] == []
    assert trust_summary["has_secret_handling"] is True
    control_summary = state["agent_control_plane"]["summary"]
    assert control_summary["control_count"] == 11
    assert control_summary["required_control_rate"] == pytest.approx(1.0)
    assert control_summary["exceeded_budget_count"] == 0
    assert control_summary["high_risk_uncontained_count"] == 0
    assert control_summary["gaps"] == []
    assert control_summary["has_kill_switch"] is True
    assert control_summary["has_drift_detection"] is True

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "agent-control-plane-optimization" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_browser_cua_optimization_example_runs_redteam_replay(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_BROWSER_CUA_OPT_EXAMPLE_KEY",
        "real-local-browser-cua-opt-key",
    )

    output_path = tmp_path / "browser-cua-optimization.json"
    junit_path = tmp_path / "browser-cua-optimization.junit.xml"
    sarif_path = tmp_path / "browser-cua-optimization.sarif.json"
    markdown_path = tmp_path / "browser-cua-optimization.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "browser_cua_optimization.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.optimization.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] >= 0.98
    assert payload["summary"]["evaluation_score"] == pytest.approx(1.0)
    assert "simulation.environments" in payload["summary"]["search_paths"]

    env_types = [
        environment["type"]
        for environment in payload["optimization"]["best_config"]["simulation"][
            "environments"
        ]
    ]
    assert env_types == ["browser_cua"]

    best_history = max(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["patch"].keys() == {"simulation.environments"}
    metrics = best_history["metrics"]
    for metric in (
        "browser_action_safety",
        "browser_action_outcome",
        "browser_grounding_quality",
        "browser_mutation_resilience",
        "browser_trace_coverage",
        "tool_selection_accuracy",
    ):
        assert metrics[metric] == pytest.approx(1.0)

    case = best_history["report"]["results"][0]
    state = case["metadata"]["environment_state"]
    assert set(state) == {"browser"}
    browser = state["browser"]
    assert browser["checkout_complete"] is True
    assert browser["order_id"] == "ord_123"
    assert browser["url"] == "https://shop.example.test/confirmation"
    assert browser["mutation_pack"]["summary"]["mutation_count"] == 2
    assert browser["action_replay"][0]["mutation_id"] == "selector_drift_checkout"
    assert browser["action_replay"][0]["selector"] == "button[data-testid='place-order-safe']"
    assert browser["action_replay"][0]["success"] is True
    assert browser["action_replay"][0]["prompt_injection_touched"] is False

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "browser-cua-redteam-optimization" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_framework_certification_optimization_example_runs_framework_evidence(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_FRAMEWORK_CERT_OPT_EXAMPLE_KEY",
        "real-local-framework-cert-opt-key",
    )

    output_path = tmp_path / "framework-certification-optimization.json"
    junit_path = tmp_path / "framework-certification-optimization.junit.xml"
    sarif_path = tmp_path / "framework-certification-optimization.sarif.json"
    markdown_path = tmp_path / "framework-certification-optimization.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "framework_certification_optimization.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.optimization.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] >= 0.98
    assert payload["summary"]["evaluation_score"] == pytest.approx(1.0)
    assert "simulation.environments" in payload["summary"]["search_paths"]

    env_types = [
        environment["type"]
        for environment in payload["optimization"]["best_config"]["simulation"][
            "environments"
        ]
    ]
    assert env_types == [
        "framework_lifecycle",
        "framework_capability",
        "framework_probe",
        "framework_portability",
    ]

    best_history = max(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["patch"].keys() == {"simulation.environments"}
    metrics = best_history["metrics"]
    for metric in (
        "framework_lifecycle_coverage",
        "framework_lifecycle_quality",
        "framework_capability_coverage",
        "framework_capability_quality",
        "framework_probe_coverage",
        "framework_probe_quality",
        "framework_portability_coverage",
        "framework_portability_quality",
        "tool_selection_accuracy",
    ):
        assert metrics[metric] == pytest.approx(1.0)

    case = best_history["report"]["results"][0]
    state = case["metadata"]["environment_state"]
    assert set(state) == {
        "framework_lifecycle_trace",
        "framework_capability_matrix",
        "framework_probe_suite",
        "framework_portability_matrix",
    }
    lifecycle = state["framework_lifecycle_trace"]["summary"]
    assert lifecycle["phase_count"] == 10
    assert lifecycle["recovered_error_count"] == 1
    capability = state["framework_capability_matrix"]["summary"]
    assert capability["supported_count"] == 9
    assert capability["missing_count"] == 0
    probe = state["framework_probe_suite"]["summary"]
    assert probe["passed_count"] == 12
    assert probe["failed_count"] == 0
    portability = state["framework_portability_matrix"]["summary"]
    assert portability["mapped_count"] == 10
    assert portability["missing_count"] == 0

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "framework-certification-optimization" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_autonomous_redteam_task_world_optimization_example_runs_full_harness(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_AUTONOMOUS_REDTEAM_OPT_EXAMPLE_KEY",
        "real-local-autonomous-redteam-opt-key",
    )

    output_path = tmp_path / "autonomous-redteam-task-world-optimization.json"
    junit_path = tmp_path / "autonomous-redteam-task-world-optimization.junit.xml"
    sarif_path = tmp_path / "autonomous-redteam-task-world-optimization.sarif.json"
    markdown_path = tmp_path / "autonomous-redteam-task-world-optimization.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "autonomous_redteam_task_world_optimization.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.optimization.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] == pytest.approx(1.0)
    assert payload["summary"]["evaluation_score"] == pytest.approx(1.0)
    assert "simulation.environments" in payload["summary"]["search_paths"]

    env_types = [
        environment["type"]
        for environment in payload["optimization"]["best_config"]["simulation"][
            "environments"
        ]
    ]
    assert env_types == [
        "structured_artifact",
        "domain_package",
        "world_attack_replay",
        "autonomy_loop",
    ]

    best_history = max(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["patch"].keys() == {"simulation.environments"}
    assert best_history["score"] == pytest.approx(1.0)
    assert {
        name: score
        for name, score in best_history["metrics"].items()
        if score < 1.0
    } == {}
    for metric in (
        "artifact_semantics_quality",
        "artifact_grounding_quality",
        "domain_package_quality",
        "world_contract_coverage",
        "world_contract_quality",
        "adversarial_resilience",
        "autonomy_loop_coverage",
        "autonomy_loop_quality",
        "tool_argument_schema",
    ):
        assert best_history["metrics"][metric] == pytest.approx(1.0)

    case = best_history["report"]["results"][0]
    state = case["metadata"]["environment_state"]
    assert set(state) == {
        "adversarial",
        "autonomy_loop",
        "domain_packages",
        "structured_artifacts",
        "world_attack_replay",
        "world_contract",
    }
    assert state["structured_artifacts"]["ids"] == ["approval_policy"]
    assert state["domain_packages"]["ids"] == ["refund_case"]
    world_summary = state["world_attack_replay"]["summary"]
    assert world_summary["world_terminal_status"] == "success"
    assert world_summary["completed_required_transition_count"] == 2
    assert world_summary["invariant_violation_count"] == 0
    assert world_summary["attack_count"] == 2
    assert world_summary["canary_count"] == 1
    assert state["autonomy_loop"]["stages_observed"] == [
        "act",
        "memory",
        "observe",
        "orient",
        "plan",
        "reflect",
        "skill",
        "status",
        "verify",
    ]

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "autonomous-redteam-task-world-optimization" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_multimodal_image_optimization_example_runs_image_evidence(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AGENT_LEARNING_MULTIMODAL_IMAGE_OPT_EXAMPLE_KEY",
        "real-local-multimodal-image-opt-key",
    )

    output_path = tmp_path / "multimodal-image-optimization.json"
    junit_path = tmp_path / "multimodal-image-optimization.junit.xml"
    sarif_path = tmp_path / "multimodal-image-optimization.sarif.json"
    markdown_path = tmp_path / "multimodal-image-optimization.md"

    exit_code = main([
        "optimize",
        str(EXAMPLES / "multimodal_image_optimization.json"),
        "--output",
        str(output_path),
        "--junit",
        str(junit_path),
        "--sarif",
        str(sarif_path),
        "--markdown",
        str(markdown_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "agent-learning.optimization.v1"
    assert payload["status"] == "passed"
    assert payload["summary"]["optimization_score"] == pytest.approx(1.0)
    assert payload["summary"]["evaluation_score"] == pytest.approx(1.0)
    assert "simulation.environments" in payload["summary"]["search_paths"]

    env_types = [
        environment["type"]
        for environment in payload["optimization"]["best_config"]["simulation"][
            "environments"
        ]
    ]
    assert env_types == ["multimodal_image"]

    best_history = max(
        payload["optimization"]["history"],
        key=lambda item: item["score"],
    )
    assert best_history["patch"].keys() == {"simulation.environments"}
    assert best_history["score"] == pytest.approx(1.0)
    assert {
        name: score
        for name, score in best_history["metrics"].items()
        if score < 1.0
    } == {}
    for metric in (
        "artifact_coverage",
        "artifact_grounding_quality",
        "artifact_semantics_quality",
        "agent_goal_accuracy",
        "multimodal_faithfulness",
        "tool_selection_accuracy",
    ):
        assert best_history["metrics"][metric] == pytest.approx(1.0)

    case = best_history["report"]["results"][0]
    state = case["metadata"]["environment_state"]
    assert state == {
        "images": {
            "ids": ["receipt_image"],
            "last_inspected": "receipt_image",
            "vision_harness": "receipt_grounding",
        }
    }

    assert "failures=\"0\"" in junit_path.read_text(encoding="utf-8")
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"] == []
    assert "multimodal-image-optimization" in markdown_path.read_text(
        encoding="utf-8"
    )
