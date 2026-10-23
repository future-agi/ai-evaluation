import runpy

import pytest

from fi.alk.harness.bundle import BundleError, export_session_bundle, load_bundle
from fi.alk.harness.credentials import discover_credentials
from fi.alk.harness.generated_runtime import (
    GeneratedRuntimeError,
    _copy_sanitized_context,
)
from fi.alk.harness.packaging import inspect_packaging
from fi.alk.harness.provision import (
    source_fingerprint,
    _declared_configuration_names,
    _url_settings,
)
from fi.alk.harness.source_data_invariants import source_files
from fi.alk.harness.sources import RepoSource


@pytest.mark.parametrize("directory", ["build", "dist", "artifacts"])
def test_application_module_survives_context_and_bundle_roundtrip(tmp_path, directory):
    source = tmp_path / "source"
    module = source / "app" / directory / "service.py"
    module.parent.mkdir(parents=True)
    module.write_text("def answer():\n    return 42\n")
    copied = tmp_path / "context"
    _copy_sanitized_context(source, copied)
    root, _ = export_session_bundle(copied, tmp_path / "session", name="app")
    load_bundle(root)
    restored = root / "services/source/app" / directory / "service.py"
    assert restored.read_bytes() == module.read_bytes()
    assert runpy.run_path(str(restored))["answer"]() == 42


@pytest.mark.parametrize("directory", ["build", "dist", "artifacts"])
def test_module_participates_in_discovery_and_fingerprint(tmp_path, directory):
    path = tmp_path / "app" / directory / "config.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        'import os\nTOKEN = os.environ["REMOTE_API_KEY"]\nURL = os.getenv("SERVICE_URL", "http://service:8080")\n'
    )
    relative = path.relative_to(tmp_path).as_posix()
    assert relative in RepoSource("app", tmp_path).briefing()
    assert relative in source_files(tmp_path)
    assert "REMOTE_API_KEY" in _declared_configuration_names(tmp_path)
    assert ("SERVICE_URL", "http://service:8080") in _url_settings(tmp_path)
    manifest = discover_credentials(tmp_path)
    assert "REMOTE_API_KEY" in {item.environment_name for item in manifest.requirements}
    before = source_fingerprint(tmp_path)
    path.write_text(path.read_text() + "VALUE = 42\n")
    assert source_fingerprint(tmp_path) != before


@pytest.mark.parametrize("directory", ["build", "dist", "artifacts"])
def test_packaging_files_are_discovered_under_ordinary_directory_names(
    tmp_path, directory
):
    folder = tmp_path / directory
    folder.mkdir()
    (folder / "Dockerfile").write_text(
        'FROM python:3.11-slim\nCOPY . /app\nWORKDIR /app\nCMD ["python", "main.py"]\n'
    )
    (folder / "main.py").write_text('print("application")\n')
    assert f"{directory}/Dockerfile" in {
        candidate.path for candidate in inspect_packaging(tmp_path).candidates
    }


@pytest.mark.parametrize("directory", ["build", "dist", "artifacts"])
def test_private_material_is_rejected_in_previously_omitted_directories(
    tmp_path, directory
):
    source = tmp_path / "source"
    file = source / "app" / directory / "config.py"
    file.parent.mkdir(parents=True)
    file.write_text(
        'KEY = "-----BEGIN PRIVATE KEY-----actual-private-material-----END PRIVATE KEY-----"\n'
    )
    with pytest.raises(GeneratedRuntimeError, match="secret_material_detected"):
        _copy_sanitized_context(source, tmp_path / "context")
    with pytest.raises(BundleError, match="secret_material_detected"):
        export_session_bundle(source, tmp_path / "session", name="app")


def test_metadata_caches_stay_excluded_and_explicit_context_selection_still_works(
    tmp_path,
):
    source = tmp_path / "source"
    for name in [".git", ".venv", "node_modules", "__pycache__", "build"]:
        path = source / name / "module.py"
        path.parent.mkdir(parents=True)
        path.write_text("VALUE = 42\n")
    output = tmp_path / "context"
    _copy_sanitized_context(source, output, exclusions=("build",))
    assert not list(output.rglob("module.py"))


def test_repeated_nested_export_omits_only_owned_session(tmp_path):
    source = tmp_path / "source"
    module = source / "artifacts" / "sibling.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 42\n")
    session = source / "artifacts" / "owned-session"
    first_root, first = export_session_bundle(source, session, name="app")
    assert (
        first_root / "services/source/artifacts/sibling.py"
    ).read_bytes() == module.read_bytes()
    (session / "run.log").write_text("new output")
    second_root, second = export_session_bundle(source, session, name="app")
    assert first.digest == second.digest
    assert not (second_root / "services/source/artifacts/owned-session").exists()
    before = source_fingerprint(source, exclude=session)
    module.write_text("VALUE = 43\n")
    assert source_fingerprint(source, exclude=session) != before


def test_nested_provision_reuses_despite_output_changes_but_not_source_changes(
    tmp_path, monkeypatch
):
    import json
    from fi.alk.harness import provision as provisioning

    source = tmp_path / "source"
    module = source / "artifacts" / "sibling.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 42\n")
    compose = source / "compose.yml"
    compose.write_text("services:\n  api:\n    image: example\n")
    output = source / "artifacts" / "owned-session"
    saved = provisioning.ProvisionedEnvironment(
        source=str(source),
        compose_file=str(compose),
        project="fagi-existing",
        services=["api"],
        running=True,
        source_fingerprint=source_fingerprint(source, exclude=output),
    )
    saved.save(output)
    calls = []

    def run(environment, *args, **kwargs):
        calls.append((environment.project, args))
        if args[:3] == ("ps", "--status", "running"):
            return "container-id"
        if "config" in args and "--format" in args:
            return json.dumps({"services": {"api": {}}})
        return ""

    monkeypatch.setattr(provisioning, "_run", run)
    monkeypatch.setattr(provisioning, "_wait_for_endpoints", lambda *args: None)
    (output / "run.log").write_text("new output")
    assert provisioning.provision(source, output).project == "fagi-existing"
    (output / "run.log").write_text("more output")
    assert provisioning.provision(source, output).project == "fagi-existing"
    module.write_text("VALUE = 43\n")
    assert provisioning.provision(source, output).project != "fagi-existing"
    assert ("fagi-existing", ("down", "--volumes", "--remove-orphans")) in calls


def test_owned_output_cannot_equal_source_and_external_exclusion_preserves_source(
    tmp_path,
):
    from fi.alk.harness.provision import ProvisionError, provision

    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("VALUE = 42\n")
    with pytest.raises(BundleError, match="bundle_session_is_source"):
        export_session_bundle(source, source, name="app")
    with pytest.raises(ProvisionError, match="provision_destination_is_source"):
        provision(source, source)
    with pytest.raises(ProvisionError, match="source_fingerprint_exclusion_is_source"):
        source_fingerprint(source, exclude=source)
    assert source_fingerprint(source, exclude=tmp_path) == source_fingerprint(source)
