import hashlib

import pytest

from fi.alk.harness.bundle import BundleError
from fi.alk.harness.certified_compose_runtime import ComposeRuntimeError, normalize_compose
from fi.alk.harness.compose_integration import validate_source
from fi.alk.harness.compose_source import source_inputs
from fi.alk.harness.provision import source_fingerprint
from fi.alk.harness.public_material import PublicMaterialAdmission


def document(**service):
    return {"services": {"app": {"image": "example/app:1", **service}}}


def fixture(source):
    path = source / "tests/fixture.py"
    path.parent.mkdir()
    path.write_text("TOKEN = 'ghp_" + "x" * 36 + "'\n")
    return path


def test_source_fingerprint_ignores_appledouble_metadata(tmp_path):
    (tmp_path / "agent.py").write_text("ready = True\n")
    expected = source_fingerprint(tmp_path)

    (tmp_path / "._agent.py").write_bytes(b"\x00\x05\x16\x07" + b"\x00" * 28)

    assert source_fingerprint(tmp_path) == expected


def test_source_fingerprint_keeps_non_appledouble_dot_underscore_files(tmp_path):
    (tmp_path / "agent.py").write_text("ready = True\n")
    expected = source_fingerprint(tmp_path)

    (tmp_path / "._agent.py").write_text("submitted_source = True\n")

    assert source_fingerprint(tmp_path) != expected


def test_image_only_needs_no_source_copy_but_keeps_full_identity(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    unused = fixture(source)
    inputs = source_inputs(source, document(), capabilities={})
    assert inputs.entries == ()
    digest = source_fingerprint(source)
    validate_source(source, paths=inputs.files, source_digest=digest)
    inputs.copy_to(tmp_path / "copy")
    assert list((tmp_path / "copy").iterdir()) == []
    unused.write_text("changed upstream fixture")
    with pytest.raises(ValueError, match="source_changed"):
        validate_source(source, paths=inputs.files, source_digest=digest)


@pytest.mark.parametrize("kind", ["bind_file", "bind_dir", "env_file", "build"])
def test_consumed_test_credentials_still_need_content_admission(tmp_path, kind):
    fixture(tmp_path)
    (tmp_path / "Dockerfile").write_text("FROM busybox\n")
    (tmp_path / ".dockerignore").write_text("tests\n")
    service = {
        "bind_file": {"volumes": ["./tests/fixture.py:/fixture.py"]},
        "bind_dir": {"volumes": ["./tests:/tests"]},
        "env_file": {"env_file": "tests/fixture.py"},
        "build": {"build": "."},
    }[kind]
    inputs = source_inputs(tmp_path, document(**service), capabilities={})
    with pytest.raises(BundleError, match="bundle_secret_material_detected"):
        validate_source(tmp_path, paths=inputs.files)


def test_declared_context_patterns_remove_test_only_build_inputs(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM busybox\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/app.ts").write_text("export const ready = true;\n")
    (tmp_path / "src/app.test.ts").write_text("const token = 'ghp_" + "x" * 36 + "';\n")
    inputs = source_inputs(
        tmp_path,
        document(build="."),
        capabilities={},
        context_excludes=("**/*.test.ts",),
    )

    validate_source(tmp_path, paths=inputs.files)
    destination = tmp_path / "copy"
    inputs.copy_to(destination)
    assert (destination / "src/app.ts").is_file()
    assert not (destination / "src/app.test.ts").exists()


def test_declared_context_pattern_cannot_remove_required_source_root(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app/Dockerfile").write_text("FROM busybox\n")

    with pytest.raises(ComposeRuntimeError, match="required_path_excluded"):
        source_inputs(
            tmp_path,
            document(build="./app"),
            capabilities={},
            context_excludes=("app",),
        )


@pytest.mark.parametrize("excluded", ["../outside", "/tmp/source", "tests/[fixture]"])
def test_context_exclusion_must_be_safe_relative_pattern(tmp_path, excluded):
    with pytest.raises(ComposeRuntimeError, match="context_exclude_invalid"):
        source_inputs(
            tmp_path,
            document(image="example/app:1"),
            capabilities={},
            context_excludes=(excluded,),
        )


@pytest.mark.parametrize("kind", ["file", "directory", "parent_alias", "env_file"])
def test_alias_chains_and_nested_targets_are_copied_with_their_closure(tmp_path, kind):
    source = tmp_path / "source"
    (source / "real").mkdir(parents=True)
    (source / "real/config.env").write_text("PUBLIC_VALUE=hello\n")
    (source / "unused.txt").write_text("unrelated")
    target = "real/config.env" if kind in {"file", "env_file"} else "real"
    (source / "second").symlink_to(target)
    (source / "first").symlink_to("second")
    if kind == "env_file":
        spec = document(env_file="first")
    elif kind == "parent_alias":
        spec = document(volumes=["./first/config.env:/config.env"])
    else:
        spec = document(volumes=["./first:/config"])
    inputs = source_inputs(source, spec, capabilities={})
    validate_source(source, paths=inputs.files)
    destination = tmp_path / "copy"
    inputs.copy_to(destination)
    assert (destination / "first").is_symlink()
    assert (destination / "second").is_symlink()
    assert (destination / "first").exists()
    assert (destination / "real/config.env").read_text() == "PUBLIC_VALUE=hello\n"
    assert not (destination / "unused.txt").exists()
    normalize_compose(spec, destination, capabilities={})


def test_directory_bind_follows_nested_symlink_and_checks_its_target(tmp_path):
    fixture(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/nested").symlink_to("../tests")
    inputs = source_inputs(tmp_path, document(volumes=["./config:/config"]), capabilities={})
    assert tmp_path / "tests/fixture.py" in inputs.files
    with pytest.raises(BundleError, match="bundle_secret_material_detected"):
        validate_source(tmp_path, paths=inputs.files)


@pytest.mark.parametrize("target", ["..", ".", "missing", "loop"])
def test_symlink_escape_cycle_or_dangling_is_rejected(tmp_path, target):
    source = tmp_path / "source"
    source.mkdir()
    (source / "loop").symlink_to(target)
    with pytest.raises((ComposeRuntimeError, RuntimeError)):
        source_inputs(source, document(volumes=["./:/source"]), capabilities={})


def test_declared_missing_resources_do_not_require_original_source_files(tmp_path):
    spec = document(env_file="runtime.env", volumes=["./workspace:/workspace"])
    spec["x-alk-config-files"] = {"runtime.env": {"format": "dotenv", "values": {"PUBLIC_VALUE": "hello"}}}
    spec["x-alk-directories"] = ["workspace"]
    inputs = source_inputs(tmp_path, spec, capabilities={})
    assert inputs.entries == ()


def test_unused_admission_remains_bound_and_validated_without_copying(tmp_path):
    path = fixture(tmp_path)
    token = b"ghp_" + b"x" * 36
    admission = PublicMaterialAdmission(
        source_digest=source_fingerprint(tmp_path), path="tests/fixture.py",
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        pattern="github_token", match_sha256=hashlib.sha256(token).hexdigest(),
        reason="Reviewed public fixture", evidence="Upstream test source",
    )
    inputs = source_inputs(tmp_path, document(), capabilities={})
    validate_source(tmp_path, [admission], paths=inputs.files)
    assert inputs.entries == ()
    path.unlink()
    with pytest.raises(ValueError, match="source_changed"):
        validate_source(tmp_path, [admission], paths=inputs.files)


def test_optional_dangling_env_file_is_not_consumed(tmp_path):
    (tmp_path / "optional.env").symlink_to("missing.env")
    spec = document(env_file=[{"path": "optional.env", "required": False}])
    assert source_inputs(tmp_path, spec, capabilities={}).entries == ()


def test_copy_preserves_source_root_access_under_private_broker_umask(tmp_path):
    import os
    import stat

    source = tmp_path / "source"
    source.mkdir(mode=0o755)
    (source / "settings.txt").write_text("hello")
    inputs = source_inputs(source, document(volumes=["./:/source:ro"]), capabilities={})
    previous = os.umask(0o077)
    try:
        inputs.copy_to(tmp_path / "copy")
    finally:
        os.umask(previous)
    assert stat.S_IMODE((tmp_path / "copy").stat().st_mode) == 0o755


def test_link_dotdot_is_resolved_after_intermediate_symlink(tmp_path):
    source = tmp_path / "source"
    (source / "real/subdir").mkdir(parents=True)
    (source / "config").mkdir()
    (source / "real/private.txt").write_text("actual target")
    (source / "private.txt").write_text("wrong target")
    (source / "hop").symlink_to("real/subdir")
    (source / "config/link").symlink_to("../hop/../private.txt")
    inputs = source_inputs(source, document(volumes=["./config:/config"]), capabilities={})
    assert source / "real/private.txt" in inputs.files
    assert source / "private.txt" not in inputs.files
    inputs.copy_to(tmp_path / "copy")
    assert (tmp_path / "copy/config/link").read_text() == "actual target"
    (source / "real/private.txt").write_text("ghp_" + "x" * 36)
    with pytest.raises(BundleError, match="bundle_secret_material_detected"):
        validate_source(source, paths=inputs.files)
