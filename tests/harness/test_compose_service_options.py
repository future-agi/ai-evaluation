import copy
import pytest
from fi.alk.harness.certified_compose_runtime import (
    normalize_compose,
    ComposeRuntimeError,
)


def test_service_names_are_scoped_dns_aliases_and_local_logs_are_bounded(tmp_path):
    document = {
        "services": {
            "api": {
                "image": "example/app:1",
                "container_name": "original-api",
                "extra_hosts": ["host.docker.internal:host-gateway"],
                "logging": {
                    "driver": "json-file",
                    "options": {"max-size": "50m", "max-file": "6"},
                },
            }
        }
    }
    before = copy.deepcopy(document)
    output = normalize_compose(document, tmp_path, capabilities={})
    service = output["services"]["api"]
    assert "container_name" not in service
    assert service["networks"] == {"default": {"aliases": ["original-api"]}}
    assert output["networks"] == {"default": {}}
    assert service["logging"]["options"] == {"max-size": "50m", "max-file": "6"}
    assert service["extra_hosts"] == {"host.docker.internal": "host-gateway"}
    assert document == before


@pytest.mark.parametrize(
    "setting",
    [
        {
            "logging": {
                "driver": "syslog",
                "options": {"syslog-address": "tcp://outside:514"},
            }
        },
        {"logging": {"driver": "json-file", "options": {"max-size": "1000g"}}},
        {"logging": {"driver": "json-file", "options": {"env": "SECRET"}}},
        {"extra_hosts": ["missing-address"]},
        {"extra_hosts": {"host": "/var/run/docker.sock"}},
        {"container_name": "${UNTRUSTED}"},
        {"networks": {"private": {"driver_opts": {"parent": "eth0"}}}},
    ],
)
def test_reject_unsafe_service_options(tmp_path, setting):
    with pytest.raises(ComposeRuntimeError):
        normalize_compose(
            {
                "services": {"api": {"image": "example/app:1", **setting}},
                "networks": {"private": {}},
            },
            tmp_path,
            capabilities={},
        )


def test_existing_network_aliases_survive_container_name_scoping(tmp_path):
    output = normalize_compose(
        {
            "services": {
                "api": {
                    "image": "example/app:1",
                    "container_name": "api-name",
                    "networks": {"private": {"aliases": ["existing"]}},
                }
            },
            "networks": {"private": {}},
        },
        tmp_path,
        capabilities={},
    )
    assert output["services"]["api"]["networks"] == {
        "private": {"aliases": ["existing", "api-name"]}
    }


def test_default_logs_bounded_and_ipv6_normalized(tmp_path):
    output = normalize_compose(
        {
            "services": {
                "api": {"image": "example/app:1", "extra_hosts": {"ipv6": "[::1]"}}
            }
        },
        tmp_path,
        capabilities={},
    )
    assert output["services"]["api"]["logging"] == {
        "driver": "json-file",
        "options": {"max-size": "20m", "max-file": "3"},
    }
    assert output["services"]["api"]["extra_hosts"] == {"ipv6": "::1"}
    with pytest.raises(ComposeRuntimeError):
        normalize_compose(
            {
                "services": {
                    "api": {
                        "image": "example/app:1",
                        "extra_hosts": {"ipv6": "[[::1]]"},
                    }
                }
            },
            tmp_path,
            capabilities={},
        )
