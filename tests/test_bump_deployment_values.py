"""Deployment values bump: what it rewrites, what it leaves, what it refuses."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OLD_BUILD = "11111111-1111-4111-8111-111111111111"
NEW_BUILD = "22222222-2222-4222-8222-222222222222"
NEW_NAME = "alk-hosted-0123456789ab"
NEW_REFERENCE = f"{NEW_NAME}:{NEW_BUILD}"
NEIGHBOUR_SECRET = "not-a-real-key"

VALUES = f"""\
backend:
  secret:
    data:
      E2B_API_KEY: "{NEIGHBOUR_SECRET}"
      # ALK_E2B_TEMPLATE_REFERENCE: "commented-out:{OLD_BUILD}"
      ALK_E2B_TEMPLATE_REFERENCE: "alk-hosted-old:{OLD_BUILD}"
      ALK_E2B_TEMPLATE_BUILD_ID: "{OLD_BUILD}"
      ALK_E2B_TEMPLATE_CPU_UNITS: "8" # sized for the old template
      ALK_E2B_TEMPLATE_MEMORY_MB: "8192"
      ALK_E2B_TEMPLATE_DISK_GB: "10"
      HARNESS_PARALLELISM_ENABLED: "true"
      HARNESS_PARALLEL_SNAPSHOT_DIGESTS: "{OLD_BUILD}"
core_backend_worker_l:
  secret:
    data:
      ALK_E2B_TEMPLATE_REFERENCE: "alk-hosted-old:{OLD_BUILD}"
      ALK_E2B_TEMPLATE_BUILD_ID: "{OLD_BUILD}"
      ALK_E2B_TEMPLATE_CPU_UNITS: "8"
      ALK_E2B_TEMPLATE_MEMORY_MB: "8192"
      ALK_E2B_TEMPLATE_DISK_GB: "10"
      HARNESS_PARALLEL_SNAPSHOT_DIGESTS: "{OLD_BUILD}"
"""

EXPECTED = {
    "ALK_E2B_TEMPLATE_REFERENCE": NEW_REFERENCE,
    "ALK_E2B_TEMPLATE_BUILD_ID": NEW_BUILD,
    "ALK_E2B_TEMPLATE_CPU_UNITS": "4",
    "ALK_E2B_TEMPLATE_MEMORY_MB": "16384",
    "ALK_E2B_TEMPLATE_DISK_GB": "22",
}


def _load():
    spec = importlib.util.spec_from_file_location(
        "bump_deployment_values", PROJECT_ROOT / "scripts" / "bump_deployment_values.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up by name while the class body runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bump = _load()


def _release(**overrides) -> dict:
    release = {
        "schema_version": "futureagi.e2b-template-release.v1",
        "alk_source_revision": "0123456789ab" + "0" * 28,
        "template_name": NEW_NAME,
        "template_build_id": NEW_BUILD,
        "template_reference": NEW_REFERENCE,
        "cpu_count": 4,
        "memory_mb": 16384,
        "verified_disk_gb": 22,
    }
    release.update(overrides)
    return release


def _write_release(tmp_path: Path, release: object) -> Path:
    path = tmp_path / "release.json"
    path.write_text(json.dumps(release), encoding="utf-8")
    return path


def _pin(tmp_path: Path, **overrides):
    return bump.load_release(_write_release(tmp_path, _release(**overrides)))


def _values_of(text: str, key: str) -> list[str]:
    return [
        line.split(":", 1)[1].split("#")[0].strip()
        for line in text.splitlines()
        if line.strip().startswith(f"{key}:")
    ]


def test_rewrites_every_template_setting_in_both_blocks(tmp_path):
    updated, changes = bump.bump_text(VALUES, _pin(tmp_path), label="values.yaml")

    for key, value in EXPECTED.items():
        assert _values_of(updated, key) == [f'"{value}"', f'"{value}"'], key
    assert len(changes) == 6


def test_only_the_named_lines_change(tmp_path):
    updated, _ = bump.bump_text(VALUES, _pin(tmp_path), label="values.yaml")

    before, after = VALUES.split("\n"), updated.split("\n")
    assert len(before) == len(after)
    changed = [
        index for index, pair in enumerate(zip(before, after)) if pair[0] != pair[1]
    ]
    named = [
        index
        for index, line in enumerate(before)
        if line.strip().startswith((*bump.TEMPLATE_KEYS, bump.PARALLEL_KEY))
    ]
    assert changed == named
    assert f'E2B_API_KEY: "{NEIGHBOUR_SECRET}"' in updated
    assert f'# ALK_E2B_TEMPLATE_REFERENCE: "commented-out:{OLD_BUILD}"' in updated


def test_keeps_indent_trailing_comment_and_crlf(tmp_path):
    crlf = VALUES.replace(
        'ALK_E2B_TEMPLATE_MEMORY_MB: "8192"', "ALK_E2B_TEMPLATE_MEMORY_MB:   8192"
    ).replace("\n", "\r\n")

    updated, _ = bump.bump_text(crlf, _pin(tmp_path), label="values.yaml")

    assert updated.count("\r\n") == crlf.count("\r\n")
    assert updated.replace("\r\n", "").count("\n") == 0
    assert (
        '      ALK_E2B_TEMPLATE_CPU_UNITS: "4" # sized for the old template\r\n'
        in updated
    )
    assert updated.count('      ALK_E2B_TEMPLATE_MEMORY_MB:   "16384"\r\n') == 2


def test_parallel_list_swaps_old_build_for_new(tmp_path):
    text = VALUES.replace(
        f'HARNESS_PARALLEL_SNAPSHOT_DIGESTS: "{OLD_BUILD}"',
        f'HARNESS_PARALLEL_SNAPSHOT_DIGESTS: "daytona-digest,{OLD_BUILD}"',
    )

    updated, _ = bump.bump_text(text, _pin(tmp_path), label="values.yaml")

    assert (
        _values_of(updated, bump.PARALLEL_KEY) == [f'"daytona-digest,{NEW_BUILD}"'] * 2
    )


def test_parallel_list_untouched_when_old_build_absent(tmp_path):
    text = VALUES.replace(
        f'HARNESS_PARALLEL_SNAPSHOT_DIGESTS: "{OLD_BUILD}"',
        "HARNESS_PARALLEL_SNAPSHOT_DIGESTS: 'daytona-digest'",
    )

    updated, changes = bump.bump_text(text, _pin(tmp_path), label="values.yaml")

    assert _values_of(updated, bump.PARALLEL_KEY) == ["'daytona-digest'"] * 2
    assert not any(bump.PARALLEL_KEY in change for change in changes)


def test_parallel_list_does_not_duplicate_new_build(tmp_path):
    text = VALUES.replace(
        f'HARNESS_PARALLEL_SNAPSHOT_DIGESTS: "{OLD_BUILD}"',
        f'HARNESS_PARALLEL_SNAPSHOT_DIGESTS: "{NEW_BUILD},{OLD_BUILD}"',
    )

    updated, _ = bump.bump_text(text, _pin(tmp_path), label="values.yaml")

    assert _values_of(updated, bump.PARALLEL_KEY) == [f'"{NEW_BUILD}"'] * 2


def test_parallel_key_may_be_absent(tmp_path):
    text = "\n".join(
        line for line in VALUES.split("\n") if bump.PARALLEL_KEY not in line
    )

    updated, _ = bump.bump_text(text, _pin(tmp_path), label="values.yaml")

    assert _values_of(updated, "ALK_E2B_TEMPLATE_BUILD_ID") == [f'"{NEW_BUILD}"'] * 2
    assert bump.PARALLEL_KEY not in updated


def test_refuses_when_a_setting_is_missing(tmp_path):
    text = VALUES.replace('      ALK_E2B_TEMPLATE_DISK_GB: "10"\n', "", 1)

    with pytest.raises(bump.BumpError, match="same number of times"):
        bump.bump_text(text, _pin(tmp_path), label="values.yaml")


def test_refuses_when_a_setting_is_missing_everywhere(tmp_path):
    text = "\n".join(
        line
        for line in VALUES.split("\n")
        if not line.strip().startswith(bump.TEMPLATE_KEYS)
    )

    with pytest.raises(bump.BumpError, match="ALK_E2B_TEMPLATE_REFERENCE x0"):
        bump.bump_text(text, _pin(tmp_path), label="values.yaml")


def test_refuses_when_blocks_disagree_on_the_old_build(tmp_path):
    other = "33333333-3333-4333-8333-333333333333"
    head, tail = VALUES.split("core_backend_worker_l:")
    text = head + "core_backend_worker_l:" + tail.replace(OLD_BUILD, other)

    with pytest.raises(bump.BumpError, match="differs between blocks"):
        bump.bump_text(text, _pin(tmp_path), label="values.yaml")


@pytest.mark.parametrize(
    "value",
    ["|", "", '"a\\"b"', "two words", "'it''s'", "[a, b]", "*anchor", "plain#note"],
)
def test_refuses_a_value_it_cannot_rewrite(tmp_path, value):
    text = VALUES.replace(
        'ALK_E2B_TEMPLATE_DISK_GB: "10"',
        f"ALK_E2B_TEMPLATE_DISK_GB: {value}".rstrip(),
        1,
    )

    with pytest.raises(bump.BumpError, match="line 10 sets ALK_E2B_TEMPLATE_DISK_GB"):
        bump.bump_text(text, _pin(tmp_path), label="values.yaml")


def test_accepts_an_empty_quoted_old_value(tmp_path):
    text = VALUES.replace(f"alk-hosted-old:{OLD_BUILD}", "").replace(
        f'ALK_E2B_TEMPLATE_BUILD_ID: "{OLD_BUILD}"', 'ALK_E2B_TEMPLATE_BUILD_ID: ""'
    )
    text = text.replace(
        f'HARNESS_PARALLEL_SNAPSHOT_DIGESTS: "{OLD_BUILD}"',
        'HARNESS_PARALLEL_SNAPSHOT_DIGESTS: ""',
    )

    updated, _ = bump.bump_text(text, _pin(tmp_path), label="values.yaml")

    assert (
        _values_of(updated, "ALK_E2B_TEMPLATE_REFERENCE") == [f'"{NEW_REFERENCE}"'] * 2
    )
    assert _values_of(updated, bump.PARALLEL_KEY) == ['""'] * 2


def test_main_writes_nothing_when_one_file_is_refused(tmp_path):
    release = _write_release(tmp_path, _release())
    good, bad = tmp_path / "us.yaml", tmp_path / "eu.yaml"
    good.write_bytes(VALUES.encode())
    bad.write_bytes(
        VALUES.replace('      ALK_E2B_TEMPLATE_DISK_GB: "10"\n', "", 1).encode()
    )

    with pytest.raises(SystemExit) as refusal:
        bump.main(["--release", str(release), str(good), str(bad)])

    assert str(refusal.value).startswith("error: ")
    assert good.read_bytes() == VALUES.encode()


def test_main_is_a_no_op_the_second_time(tmp_path, capsys):
    release = _write_release(tmp_path, _release())
    values = tmp_path / "values.yaml"
    values.write_bytes(VALUES.encode())

    assert bump.main(["--release", str(release), str(values)]) == 0
    first = values.read_bytes()
    assert first != VALUES.encode()
    capsys.readouterr()
    assert bump.main(["--release", str(release), str(values)]) == 0

    assert values.read_bytes() == first
    assert "already pinned" in capsys.readouterr().out


def test_main_keeps_crlf_line_endings(tmp_path):
    release = _write_release(tmp_path, _release())
    values = tmp_path / "values.yaml"
    values.write_bytes(VALUES.replace("\n", "\r\n").encode())

    assert bump.main(["--release", str(release), str(values)]) == 0

    written = values.read_bytes()
    assert NEW_REFERENCE.encode() in written
    assert written.count(b"\r\n") == written.count(b"\n") == VALUES.count("\n")


@pytest.mark.parametrize("content", ["", "not json", "{"])
def test_release_that_is_not_json_is_refused(tmp_path, content):
    path = tmp_path / "release.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(bump.BumpError, match="cannot read the template release"):
        bump.load_release(path)


def test_release_must_pin_its_own_build(tmp_path):
    other = "33333333-3333-4333-8333-333333333333"

    with pytest.raises(bump.BumpError, match="template_reference"):
        _pin(tmp_path, template_reference=f"{NEW_NAME}:{other}")


@pytest.mark.parametrize(
    "release",
    [
        _release(schema_version="futureagi.e2b-template-release.v2"),
        _release(
            template_build_id="not-a-uuid", template_reference=f"{NEW_NAME}:not-a-uuid"
        ),
        _release(
            template_build_id=NEW_BUILD.replace("2", "A"),
            template_reference=f"{NEW_NAME}:{NEW_BUILD.replace('2', 'A')}",
        ),
        _release(cpu_count=0),
        _release(memory_mb="8192"),
        _release(verified_disk_gb=True),
        _release(
            template_name="alk/hosted", template_reference=f"alk/hosted:{NEW_BUILD}"
        ),
        [_release()],
    ],
)
def test_release_validation(tmp_path, release):
    with pytest.raises(bump.BumpError):
        bump.load_release(_write_release(tmp_path, release))


def test_output_never_shows_file_contents(tmp_path, capsys):
    release = _write_release(tmp_path, _release())
    good, bad = tmp_path / "us.yaml", tmp_path / "eu.yaml"
    good.write_bytes(VALUES.encode())
    bad.write_bytes(
        VALUES.replace(
            'ALK_E2B_TEMPLATE_DISK_GB: "10"', "ALK_E2B_TEMPLATE_DISK_GB: |", 1
        ).encode()
    )

    assert bump.main(["--release", str(release), str(good)]) == 0
    with pytest.raises(SystemExit) as refusal:
        bump.main(["--release", str(release), str(bad)])

    captured = capsys.readouterr()
    shown = captured.out + captured.err + str(refusal.value)
    assert NEW_REFERENCE in shown
    for hidden in (
        NEIGHBOUR_SECRET,
        OLD_BUILD,
        "alk-hosted-old",
        "sized for the old template",
    ):
        assert hidden not in shown
