import pytest
import yaml

from fi.alk.harness.bundle_v2 import CapabilityV2
from fi.alk.harness.certified_compose_runtime import (
    ComposeRuntimeError,
    normalize_compose,
)
from fi.alk.harness.contract import Runtime
from fi.alk.harness.runtime_selection import RuntimeSelectionError, select_runtime


def declaration(**changes):
    return {
        "execution": "compose",
        "compose_file": "compose.yaml",
        "service": "chat",
        "additional_services": ["search", "queue"],
        **changes,
    }


def source(tmp_path, **overrides):
    services = {
        "chat": {"image": "chat:1", "depends_on": ["db"]},
        "db": {"image": "postgres:16"},
        "search": {
            "image": "search:1",
            "profiles": ["search"],
            "depends_on": ["index"],
        },
        "index": {"image": "index:1"},
        "queue": {"image": "queue:1"},
        "optional": {"image": "optional:1"},
        **overrides,
    }
    (tmp_path / "compose.yaml").write_text(yaml.safe_dump({"services": services}))
    return tmp_path


@pytest.mark.parametrize("roots", [["search", "queue"], ["queue", "search"]])
def test_explicit_roots_keep_each_dependency_closure_and_primary(tmp_path, roots):
    selection = select_runtime(source(tmp_path), declaration(additional_services=roots))
    assert selection.control_service == "chat"
    assert set(selection.services) == {"chat", "db", "search", "index", "queue"}
    order = list(selection.services)
    assert order.index("index") < order.index("search")
    assert order.index("db") < order.index("chat")


def test_omitted_sibling_is_not_implicitly_selected(tmp_path):
    selection = select_runtime(source(tmp_path), declaration(additional_services=[]))
    assert set(selection.services) == {"chat", "db"}


@pytest.mark.parametrize(
    "changes",
    [
        {"execution": "native", "command": ["python", "app.py"]},
        {"compose_file": ""},
        {"service": ""},
        {"additional_services": ["queue", "queue"]},
        {"additional_services": ["chat"]},
        {"additional_services": ["../queue"]},
        {"additional_services": [""]},
        {"additional_services": ["q" * 129]},
        {"additional_services": [f"q{i}" for i in range(33)]},
    ],
)
def test_invalid_additional_roots_rejected(changes):
    with pytest.raises(ValueError):
        Runtime.model_validate(declaration(**changes))


def test_unknown_additional_root_fails_closed(tmp_path):
    with pytest.raises(RuntimeSelectionError, match="compose_dependency_missing"):
        select_runtime(source(tmp_path), declaration(additional_services=["missing"]))


def test_additional_root_cycle_fails_closed(tmp_path):
    with pytest.raises(RuntimeSelectionError, match="compose_dependency_cycle"):
        select_runtime(
            source(tmp_path, index={"image": "index:1", "depends_on": ["search"]}),
            declaration(),
        )


@pytest.mark.parametrize(
    "danger",
    [{"privileged": True}, {"volumes": ["/var/run/docker.sock:/var/run/docker.sock"]}],
)
def test_additional_service_cannot_bypass_runtime_isolation(tmp_path, danger):
    selection = select_runtime(
        source(tmp_path, queue={"image": "queue:1", **danger}), declaration()
    )
    for service in selection.services.values():
        service.pop("profiles", None)
    with pytest.raises(
        ComposeRuntimeError, match="fields_forbidden|host_bind_forbidden"
    ):
        normalize_compose(
            {"services": selection.services},
            tmp_path,
            capabilities={
                "chat": CapabilityV2(
                    protocol="http", service="chat", container_port=8080
                )
            },
        )
