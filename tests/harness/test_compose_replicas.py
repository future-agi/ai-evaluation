import copy

import pytest

from fi.alk.harness.certified_compose_runtime import (
    ComposeRuntimeError,
    normalize_compose,
)
from fi.alk.harness.runtime_selection import select_runtime


@pytest.mark.parametrize("replicas", [1, "1"])
def test_single_replica_preserved_without_source_mutation(tmp_path, replicas):
    document = {
        "services": {
            "app": {"image": "example/app:1", "deploy": {"replicas": replicas}}
        }
    }
    original = copy.deepcopy(document)
    result = normalize_compose(document, tmp_path, capabilities={})
    assert result["services"]["app"]["deploy"] == {"replicas": 1}
    assert document == original


@pytest.mark.parametrize(
    "replicas",
    [0, "0", 2, "2", -1, 10000, True, False, 1.0, None, [], {}, "01", "${COUNT:-1}"],
)
def test_zero_multi_or_invalid_replicas_fail_explicitly(tmp_path, replicas):
    with pytest.raises(
        ComposeRuntimeError,
        match="compose_replicas_unsupported: app: exactly one replica",
    ):
        normalize_compose(
            {
                "services": {
                    "app": {"image": "example/app:1", "deploy": {"replicas": replicas}}
                }
            },
            tmp_path,
            capabilities={},
        )


@pytest.mark.parametrize(
    "deploy",
    [
        None,
        [],
        "1",
        {},
        {"mode": "global"},
        {"replicas": 1, "placement": {"constraints": ["node.role == manager"]}},
        {
            "replicas": 1,
            "resources": {"reservations": {"devices": [{"capabilities": ["gpu"]}]}},
        },
        {"replicas": 1, "restart_policy": {"condition": "any"}},
    ],
)
def test_other_deployment_fields_stay_forbidden(tmp_path, deploy):
    with pytest.raises(
        ComposeRuntimeError, match="compose_deploy_fields_unsupported: app"
    ):
        normalize_compose(
            {"services": {"app": {"image": "example/app:1", "deploy": deploy}}},
            tmp_path,
            capabilities={},
        )


@pytest.mark.parametrize("count", [None, "1", "0", "2"])
def test_selected_dependency_replica_default_and_override(tmp_path, count):
    source = """services:
  app:
    image: example/app:1
    depends_on: [store]
  store:
    image: example/store:1
    deploy:
      replicas: ${STORE_REPLICAS:-1}
"""
    path = tmp_path / "compose.yaml"
    path.write_text(source)
    selected = select_runtime(
        tmp_path,
        {"execution": "compose", "compose_file": "compose.yaml", "service": "app"},
        validate_environment_dependencies=False,
        public_environment={} if count is None else {"STORE_REPLICAS": count},
    )
    document = {"services": selected.services}
    if count in ("0", "2"):
        with pytest.raises(
            ComposeRuntimeError, match="compose_replicas_unsupported: store"
        ):
            normalize_compose(document, tmp_path, capabilities={})
    else:
        output = normalize_compose(document, tmp_path, capabilities={})
        assert set(output["services"]) == {"app", "store"}
        assert output["services"]["store"]["deploy"] == {"replicas": 1}
    assert path.read_text() == source
