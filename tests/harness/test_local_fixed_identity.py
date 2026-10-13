import asyncio
from types import SimpleNamespace

import pytest

from fi.alk.harness import local_certified_runtime as local
from fi.alk.harness.process_runtime import (
    default_user_resolver,
    fixed_sandbox_user_resolver,
)


@pytest.mark.parametrize("fixed_user", [False, True])
def test_local_execution_uses_selected_process_identity(tmp_path, monkeypatch, fixed_user):
    monkeypatch.setenv("ALK_FIXED_SANDBOX_USER", "1" if fixed_user else "0")
    monkeypatch.delenv("ALK_SANDBOX_UPLOAD_ROOT", raising=False)
    authoring = tmp_path / "authoring"
    (authoring / "generic-harness" / "certified-bundle").mkdir(parents=True)

    async def validated(*args, **kwargs):
        return None

    monkeypatch.setattr(local, "validate_and_repair", validated)
    monkeypatch.setattr(local, "load_bundle_v2", lambda _: object())
    monkeypatch.setattr(local, "_make_local_seed_files_readable", lambda _: None)

    class ObservedIdentity(Exception):
        pass

    class Provider:
        def __init__(self, **kwargs):
            expected = fixed_sandbox_user_resolver if fixed_user else default_user_resolver
            assert kwargs["user_resolver"] is expected
            raise ObservedIdentity

    monkeypatch.setattr(local, "ProcessRuntimeProvider", Provider)
    job = SimpleNamespace(job_id="identity", agent=SimpleNamespace(secret_refs={}))
    with pytest.raises(ObservedIdentity):
        asyncio.run(local.run_local_certified(job, source=tmp_path, authoring=authoring))
