import asyncio
import subprocess
import sys

from fi.alk.harness.process_runtime import (
    ProcessRuntimeProvider,
    PopenProcess,
    SpawnedWorldProcess,
)


def test_close_preserves_safe_real_process_output_and_cleans_work(tmp_path):
    work = tmp_path / "work"
    log = work / "worlds" / "w0" / "agent" / "process.log"
    log.parent.mkdir(parents=True)
    with log.open("wb") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                'import sys; print("stdout marker secret123"); print("stderr marker", file=sys.stderr)',
            ],
            stdout=stream,
            stderr=stream,
        )
        process.wait(timeout=5)
    provider = ProcessRuntimeProvider(
        generic_artifact_root=tmp_path / "artifacts", secrets_path=tmp_path / "secrets"
    )
    provider._secret_values = {"api_key": "secret123"}
    provider._secrets_loaded = True
    provider._world_handles = {
        0: {"agent": SpawnedWorldProcess("agent", PopenProcess(process, log), 0, 0)}
    }
    asyncio.run(provider.close(work_directory=work))
    retained = list((tmp_path / "artifacts").rglob("*.log"))
    assert retained
    text = "\n".join(path.read_text() for path in retained)
    assert "stdout marker" in text and "stderr marker" in text
    assert "secret123" not in text
    assert not (work / "worlds").exists()
    asyncio.run(provider.close(work_directory=work))
    assert list((tmp_path / "artifacts").rglob("*.log")) == retained


def test_retention_drops_split_credentials_and_rejects_symlinks(tmp_path):
    import pytest
    from fi.alk.harness.process_diagnostics import retain_process_log

    work = tmp_path / "work"
    work.mkdir()
    log = work / "process.log"
    secret = "start" + "z" * 100000 + "finish"
    log.write_text(secret + "\nfinal output\nAuthorization: Bearer unknown\n")
    retain_process_log(
        log, work=work, destination=tmp_path / "artifacts", secrets={"token": secret}
    )
    body = next((tmp_path / "artifacts").rglob("*.log")).read_text()
    assert "final output" in body
    assert "finish" not in body and "unknown" not in body
    assert len(body.encode()) <= 32768
    log.unlink()
    log.symlink_to(tmp_path / "private")
    (tmp_path / "private").write_text("must not read")
    with pytest.raises(OSError):
        retain_process_log(
            log, work=work, destination=tmp_path / "artifacts", secrets={}
        )


def test_retention_write_failure_does_not_prevent_cleanup(tmp_path, monkeypatch):
    from fi.alk.harness import process_diagnostics

    work = tmp_path / "work"
    log = work / "worlds" / "w0" / "agent" / "process.log"
    log.parent.mkdir(parents=True)
    log.write_text("log")
    provider = ProcessRuntimeProvider(secrets_path=tmp_path / "secrets")

    def fail(*args, **kwargs):
        raise OSError("no space")

    monkeypatch.setattr(process_diagnostics, "retain_process_log", fail)
    asyncio.run(provider.close(work_directory=work))
    assert not (work / "worlds").exists()


def test_reset_retains_generation_before_replacement(tmp_path, monkeypatch):
    from fi.alk.harness import process_runtime as pr
    from test_process_runtime import _manifest, _spawn_context

    manifest = _manifest()
    context = _spawn_context(manifest, bundle_dir=tmp_path, work_directory=tmp_path)
    provider = ProcessRuntimeProvider(
        generic_artifact_root=tmp_path / "artifacts", secrets_path=tmp_path / "secrets"
    )
    provider._diagnostic_work_directory = tmp_path
    provider._secrets_loaded = True
    from dataclasses import replace

    context = replace(context, retain_process_log=provider._retain_process_log)
    log = tmp_path / "worlds" / "w0" / "agent" / "process.log"
    log.parent.mkdir(parents=True)
    log.write_text("first generation\n")
    with log.open("ab") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                'import sys; print("reset stderr", file=sys.stderr)',
            ],
            stdout=stream,
            stderr=stream,
        )
        process.wait(timeout=5)
    handle = SpawnedWorldProcess("agent", PopenProcess(process, log), 0, 0)

    class StoppedAfterRetention(Exception):
        pass

    def seal(*args, **kwargs):
        assert process.poll() is not None
        assert any(
            "reset stderr" in path.read_text()
            for path in (tmp_path / "artifacts").rglob("*.log")
        )
        raise StoppedAfterRetention

    monkeypatch.setattr(pr, "_seal_world_store", seal)
    import pytest

    for _ in range(2):
        with pytest.raises(StoppedAfterRetention):
            pr.reset_world(
                manifest,
                0,
                context=context,
                baseline=pr.BuildOutput(bundle_digest=manifest.digest, stores=[]),
                job_shared_handles={},
                existing_handles={"agent": handle},
            )
    assert len(list((tmp_path / "artifacts").rglob("*.log"))) == 2


def test_close_captures_output_flushed_during_termination(tmp_path):
    import time

    work = tmp_path / "work"
    log = work / "worlds" / "w0" / "agent" / "process.log"
    log.parent.mkdir(parents=True)
    script = 'import signal,sys,time; signal.signal(signal.SIGTERM, lambda *_: (print("termination marker", flush=True), sys.exit(0))); print("ready", flush=True); time.sleep(60)'
    with log.open("wb") as stream:
        process = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=stream,
            stderr=stream,
            start_new_session=True,
        )
    try:
        deadline = time.monotonic() + 5
        while "ready" not in log.read_text() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "ready" in log.read_text()
        provider = ProcessRuntimeProvider(secrets_path=tmp_path / "secrets")
        provider._secrets_loaded = True
        provider._world_handles = {
            0: {"agent": SpawnedWorldProcess("agent", PopenProcess(process, log), 0, 0)}
        }
        asyncio.run(provider.close(work_directory=work))
        assert process.poll() == 0
        assert any(
            "termination marker" in path.read_text()
            for path in (work / "diagnostics").rglob("*.log")
        )
        assert not (work / "worlds").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_current_attempt_evidence_excludes_old_symlink_and_large_logs(tmp_path):
    from fi.alk.harness.process_diagnostics import retained_process_evidence

    old = tmp_path / "old.log"
    old.write_text("OLD")
    current = tmp_path / "current.log"
    current.write_text("missing declared dependency token-secret")
    (tmp_path / "link.log").symlink_to(old)
    (tmp_path / "large.log").write_text("x" * 32769)
    evidence = retained_process_evidence(tmp_path, {old}, {"key": "token-secret"})
    assert "missing declared dependency" in evidence
    assert "token-secret" not in evidence
    assert "OLD" not in evidence
    assert "large.log" not in evidence
    assert "link.log" not in evidence
    assert len(evidence) <= 12000


def test_target_evidence_excludes_build_sidecar_and_old_target_logs(tmp_path):
    from fi.alk.harness.process_diagnostics import retained_process_evidence

    token = "a" * 32
    for name, content in (
        (f"-{token}.log", "BUILD install buildpkg[extra]"),
        (f"worlds-w0-db-{token}.log", "SIDECAR install dbpkg[extra]"),
        (f"worlds-w0-web-{token}.log", "TARGET install webpkg[extra]"),
        (f"worlds-w1-web-{token}.log", "OLD TARGET"),
    ):
        (tmp_path / name).write_text(content)
    evidence = retained_process_evidence(
        tmp_path, {tmp_path / f"worlds-w1-web-{token}.log"}, {}, process_name="web"
    )
    assert "TARGET install webpkg[extra]" in evidence
    assert "BUILD" not in evidence
    assert "SIDECAR" not in evidence
    assert "OLD TARGET" not in evidence
