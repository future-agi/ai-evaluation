import json
import os
import subprocess
import uuid

import pytest

from fi.alk.harness.certified_compose_runtime import (
    ComposeRuntimeError,
    normalize_compose,
)


def topology(tmpfs):
    return {"services": {"cache": {"image": "alpine:3.21", "tmpfs": tmpfs}}}


@pytest.mark.parametrize(
    "tmpfs,expected",
    [
        ("/data", ["/data"]),
        (["/data"], ["/data"]),
        (["/tmp:mode=1777,size=512m"], ["/tmp:mode=1777,size=512m"]),
        (["/cache:size=1024,mode=0700"], ["/cache:size=1024,mode=0700"]),
    ],
)
def test_source_tmpfs_preserves_container_semantics(tmp_path, tmpfs, expected):
    document = topology(tmpfs)
    result = normalize_compose(document, tmp_path, capabilities={})
    assert result["services"]["cache"]["tmpfs"] == expected
    assert result["services"]["cache"]["volumes"] == []
    assert document == topology(tmpfs)


@pytest.mark.parametrize(
    "entry",
    [
        "/",
        "relative",
        "/data/../escape",
        "/data//nested",
        "/data/",
        "/data\\escape",
        "/data\x00",
        "/data\n",
        "/${PATH}",
        "/data:",
        "/data:rw",
        "/data:uid=0",
        "/data:mode=9999",
        "/data:mode=1777,mode=0700",
        "/data:size=0",
        "/data:size=-1",
        "/data:size=512m,unknown=1",
        "/data:size=1m:rw",
        ["/data", "/data"],
        ["/data", "/data/child"],
        [None],
        {"/data": {}},
        [f"/path{i}" for i in range(33)],
    ],
)
def test_invalid_tmpfs_rejected(tmp_path, entry):
    with pytest.raises(ComposeRuntimeError):
        normalize_compose(topology(entry), tmp_path, capabilities={})


@pytest.mark.parametrize("secret", [False, True])
def test_tmpfs_cannot_hide_other_mounts(tmp_path, secret):
    document = topology("/data")
    if secret:
        document["services"]["cache"]["x-alk-secret-files"] = [
            {"alias": "KEY", "destination": "/data/key"}
        ]
    else:
        document["volumes"] = {"state": {}}
        document["services"]["cache"]["volumes"] = ["state:/data/nested"]
    with pytest.raises(ComposeRuntimeError, match="mount_overlap"):
        normalize_compose(document, tmp_path, capabilities={})


def test_real_docker_tmpfs_mount_and_reset(tmp_path):
    image = os.environ.get("ALK_TEST_COMPOSE_TMPFS_IMAGE")
    if not image:
        pytest.skip("requires explicitly selected cached Docker image")
    assert image.startswith("sha256:") or "@sha256:" in image
    document = topology(["/data", "/scratch:mode=1777,size=1m"])
    document["services"]["cache"].update(image=image, command=["sleep", "300"])
    normalized = normalize_compose(document, tmp_path, capabilities={})
    config = tmp_path / "compose.json"
    config.write_text(json.dumps(normalized))
    project = "alk-tmpfs-" + uuid.uuid4().hex[:12]
    command = ["docker", "compose", "-p", project, "-f", str(config)]

    def compose(*args):
        return subprocess.run(
            [*command, *args], check=True, capture_output=True, text=True, timeout=60
        ).stdout

    try:
        compose("up", "-d", "--pull", "never")
        mounts = compose("exec", "-T", "cache", "cat", "/proc/mounts")
        assert any(
            row.split()[1:3] == ["/data", "tmpfs"] for row in mounts.splitlines()
        )
        scratch = next(
            row for row in mounts.splitlines() if row.split()[1] == "/scratch"
        )
        assert scratch.split()[2] == "tmpfs"
        assert "size=1024k" in scratch
        assert (
            compose("exec", "-T", "cache", "stat", "-c", "%a", "/scratch").strip()
            == "1777"
        )
        compose("exec", "-T", "cache", "sh", "-c", "echo isolated > /data/proof")
        assert (
            compose("exec", "-T", "cache", "cat", "/data/proof").strip() == "isolated"
        )
        compose("down", "--volumes", "--remove-orphans")
        compose("up", "-d", "--pull", "never")
        compose("exec", "-T", "cache", "test", "!", "-e", "/data/proof")
    finally:
        compose("down", "--volumes", "--remove-orphans")
