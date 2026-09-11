import json

import pytest

from fi.alk.harness.artifacts import (
    ARTIFACT_MANIFEST_NAME,
    ArtifactIntegrityError,
    load_artifact_manifest,
    seal_artifacts,
)


def _case(root, name="book-a-ride", *, recording=False):
    case = root / "runs" / "run-1" / name
    case.mkdir(parents=True)
    result = {
        "scenario": name,
        "ended": "completed",
        "passed": True,
        "transcript": "user: Ride please\nassistant: Your ride is booked.",
    }
    if recording:
        track = case / "mixed.wav"
        track.write_bytes(b"RIFF-real-call-evidence")
        result.update(recording="room-1", tracks=[{"path": "mixed.wav"}])
    (case / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (case / "transcript.txt").write_text(result["transcript"], encoding="utf-8")
    (case / "agent-tool-calls.jsonl").write_text(
        json.dumps({"name": "book_ride", "ok": True}) + "\n", encoding="utf-8"
    )
    return case


def test_seals_and_verifies_complete_content_addressed_evidence(tmp_path):
    _case(tmp_path, recording=True)

    manifest = seal_artifacts(tmp_path, run_id="run-1", expected_scenarios=1)

    assert manifest.digest.startswith("sha256:")
    assert manifest.scenarios[0].recording_paths == ["runs/run-1/book-a-ride/mixed.wav"]
    assert manifest.scenarios[0].tool_trace_path.endswith("agent-tool-calls.jsonl")
    assert (tmp_path / ARTIFACT_MANIFEST_NAME).is_file()
    assert load_artifact_manifest(tmp_path).digest == manifest.digest


def test_verification_detects_artifact_tampering(tmp_path):
    case = _case(tmp_path)
    seal_artifacts(tmp_path, run_id="run-1", expected_scenarios=1)
    (case / "transcript.txt").write_text("changed", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="artifact_changed"):
        load_artifact_manifest(tmp_path)


def test_rejects_missing_transcript_and_incomplete_scenario_set(tmp_path):
    case = _case(tmp_path)
    (case / "transcript.txt").unlink()
    result = json.loads((case / "result.json").read_text(encoding="utf-8"))
    result["transcript"] = ""
    (case / "result.json").write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="transcript_missing"):
        seal_artifacts(tmp_path, run_id="run-1")

    (case / "result.json").write_text(
        json.dumps({**result, "transcript": "user: hello"}), encoding="utf-8"
    )
    with pytest.raises(ArtifactIntegrityError, match="scenario_count_mismatch"):
        seal_artifacts(tmp_path, run_id="run-1", expected_scenarios=2)


@pytest.mark.parametrize("ended", ["finished", "gave-up", "ran-out-of-turns"])
def test_accepts_every_terminal_chat_outcome_without_changing_its_grade(
    tmp_path, ended
):
    case = _case(tmp_path)
    result_path = case / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(ended=ended, passed=ended == "finished")
    result_path.write_text(json.dumps(result), encoding="utf-8")

    manifest = seal_artifacts(tmp_path, run_id="run-1", expected_scenarios=1)

    assert manifest.scenarios[0].canonical_status == ended
    persisted = json.loads(result_path.read_text(encoding="utf-8"))
    assert persisted["passed"] is (ended == "finished")


def test_rejects_a_chat_result_that_has_not_reached_a_terminal_outcome(tmp_path):
    case = _case(tmp_path)
    result_path = case / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["ended"] = ""
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="result_not_terminal"):
        seal_artifacts(tmp_path, run_id="run-1", expected_scenarios=1)


def test_rejects_recording_claim_without_durable_track(tmp_path):
    case = _case(tmp_path)
    result_path = case / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["recording"] = "room-with-no-export"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="recording_evidence_missing"):
        seal_artifacts(tmp_path, run_id="run-1")


def test_rejects_secret_files_and_secret_material(tmp_path):
    _case(tmp_path)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=hidden", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="secret_file_forbidden"):
        seal_artifacts(tmp_path, run_id="run-1")

    (tmp_path / ".env").unlink()
    (tmp_path / "debug.log").write_text(
        "token=sk-abcdefghijklmnopqrstuv", encoding="utf-8"
    )
    with pytest.raises(ArtifactIntegrityError, match="secret_material_detected"):
        seal_artifacts(tmp_path, run_id="run-1")


def test_rejects_symlinks_and_total_size_over_limit(tmp_path):
    case = _case(tmp_path)
    (tmp_path / "linked-result.json").symlink_to(case / "result.json")
    with pytest.raises(ArtifactIntegrityError, match="symlink_forbidden"):
        seal_artifacts(tmp_path, run_id="run-1")

    (tmp_path / "linked-result.json").unlink()
    with pytest.raises(ArtifactIntegrityError, match="size_limit_exceeded"):
        seal_artifacts(tmp_path, run_id="run-1", max_bytes=10)
