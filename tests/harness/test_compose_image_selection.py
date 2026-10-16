import json

import pytest
import yaml

from fi.alk.harness.contract import AgentContract, Runtime
from fi.alk.harness.runtime_repair import (
    RuntimePlanPatch,
    _contract_hash,
    apply_runtime_plan_patch,
)
from fi.alk.harness.runtime_selection import RuntimeSelectionError, select_runtime


def declaration(**changes):
    return {
        "execution": "compose",
        "compose_file": "compose.yaml",
        "service": "chat",
        "interface": {"kind": "http", "port": 8080, "path": "/chat"},
        "compose_image_services": ["chat"],
        **changes,
    }


def source(root, **overrides):
    services = {
        "chat": {
            "build": ".",
            "image": "example/chat:1",
            "depends_on": ["worker"],
            "entrypoint": ["server"],
            "environment": {"MODE": "chat"},
            "volumes": ["data:/data"],
        },
        "worker": {"build": "./worker", "image": "example/worker:1"},
        "unselected": {"image": "example/optional:1"},
        **overrides,
    }
    (root / "compose.yaml").write_text(yaml.safe_dump({"services": services}))
    return root


def test_source_image_selection_preserves_source_and_other_builds(tmp_path):
    source(tmp_path)
    before = (tmp_path / "compose.yaml").read_bytes()
    default = select_runtime(tmp_path, declaration(compose_image_services=[]))
    selected = select_runtime(tmp_path, declaration())
    assert set(selected.services) == {"chat", "worker"}
    assert selected.services["chat"] == {
        key: value for key, value in default.services["chat"].items() if key != "build"
    }
    assert selected.services["worker"] == default.services["worker"]
    assert selected.document["services"]["chat"]["build"] == "."
    assert (tmp_path / "compose.yaml").read_bytes() == before


def test_image_selection_ignores_only_selected_build_configuration(tmp_path):
    build = {"context": ".", "args": {"TOKEN": "${BUILD_TOKEN:?build only}"}}
    source(tmp_path, chat={"image": "example/chat:1", "build": build})
    selected = select_runtime(tmp_path, declaration())
    assert selected.services["chat"]["image"] == "example/chat:1"
    assert "build" not in selected.services["chat"]

    source(
        tmp_path,
        worker={"image": "example/worker:1", "build": build},
    )
    with pytest.raises(
        RuntimeSelectionError, match="compose_structural_configuration_missing: BUILD_TOKEN"
    ):
        select_runtime(tmp_path, declaration())


@pytest.mark.parametrize(
    "changes",
    [
        {"execution": "native", "command": ["python", "app.py"]},
        {"compose_file": ""},
        {"service": ""},
        {"compose_image_services": ["chat", "chat"]},
        {"compose_image_services": ["../chat"]},
        {"compose_image_services": [""]},
        {"compose_image_services": ["a" * 129]},
        {"compose_image_services": [f"service{i}" for i in range(65)]},
    ],
)
def test_invalid_image_selection_rejected(changes):
    with pytest.raises(ValueError):
        Runtime.model_validate(declaration(**changes))


@pytest.mark.parametrize("name", ["missing", "unselected"])
def test_image_selection_cannot_expand_source_closure(tmp_path, name):
    with pytest.raises(RuntimeSelectionError, match="compose_image_service_unselected"):
        select_runtime(source(tmp_path), declaration(compose_image_services=[name]))


@pytest.mark.parametrize("image", [None, "", " ", 123, "${MISSING}", "$${MISSING}"])
def test_image_selection_needs_resolved_source_image(tmp_path, image):
    with pytest.raises(RuntimeSelectionError):
        select_runtime(
            source(tmp_path, chat={"build": ".", "image": image}), declaration()
        )


def test_image_selection_resolves_source_variable_and_image_only_is_idempotent(
    tmp_path,
):
    source(tmp_path, chat={"image": "${RELEASE}"})
    selected = select_runtime(
        tmp_path, declaration(), public_environment={"RELEASE": "example/chat:2"}
    )
    assert selected.services["chat"]["image"] == "example/chat:2"
    assert "build" not in selected.services["chat"]
    assert (
        select_runtime(
            tmp_path, declaration(), public_environment={"RELEASE": "example/chat:2"}
        )
        == selected
    )


def test_included_image_selection_retains_relative_mounts(tmp_path):
    (tmp_path / "parts").mkdir()
    (tmp_path / "compose.yaml").write_text("include: [parts/chat.yaml]\nservices: {}\n")
    included = tmp_path / "parts/chat.yaml"
    included.write_text(
        "services:\n  chat:\n    image: example/chat:1\n    build: .\n    volumes: [./data:/data]\n"
    )
    before = included.read_bytes()
    selected = select_runtime(tmp_path, declaration())
    assert "build" not in selected.services["chat"]
    assert selected.services["chat"]["volumes"] == ["./parts/data:/data"]
    assert included.read_bytes() == before


def test_runtime_repair_retains_source_image_selection(tmp_path, monkeypatch):
    source(tmp_path)
    monkeypatch.setenv("ALK_TRUSTED_COMPOSE_SOCKET", "/private/alk/docker.sock")
    runtime = Runtime.model_validate(declaration(compose_image_services=[]))
    contract = AgentContract(agent="agent", runtime=runtime)
    path = tmp_path / "contract.json"
    path.write_text(contract.model_dump_json())
    patch = RuntimePlanPatch(
        base_contract_hash=_contract_hash(contract),
        runtime=Runtime.model_validate(declaration()),
        evidence_paths=("compose.yaml",),
        diagnostic_codes=("generated_runtime_plan_invalid",),
        summary="Select the source-declared published image.",
    )
    repaired = apply_runtime_plan_patch(
        tmp_path,
        path,
        patch,
        allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
    )
    assert repaired.runtime.compose_image_services == ["chat"]
    assert json.loads(path.read_text())["runtime"]["compose_image_services"] == ["chat"]


def test_image_selection_accepts_dependencies_and_explicit_roots(tmp_path):
    selected = select_runtime(
        source(tmp_path),
        declaration(
            compose_image_services=["worker", "unselected"],
            additional_services=["unselected"],
        ),
    )
    assert "build" in selected.services["chat"]
    assert "build" not in selected.services["worker"]
    assert selected.services["unselected"]["image"] == "example/optional:1"
