import hashlib
import json

import pytest


@pytest.fixture
def resolved_compose_images(monkeypatch):
    from fi.alk.harness.certified_compose_runtime import CertifiedComposeRuntimeProvider
    from fi.alk.harness.compose_broker import ComposeBrokerClient
    from fi.alk.harness.compose_image_lock import ResolvedComposeImage, image_repository

    calls = []

    def resolve(self, *, manifest, document, runtime, source):
        calls.append(document)
        return [
            ResolvedComposeImage(
                service=name,
                declared_reference=service["image"],
                immutable_reference=image_repository(service["image"])
                + "@sha256:"
                + hashlib.sha256(service["image"].encode()).hexdigest(),
                platform="linux/amd64",
                correspondence="dependency",
            )
            for name, service in json.loads(document)["services"].items()
            if not service.get("build")
        ]

    monkeypatch.setattr(CertifiedComposeRuntimeProvider, "resolve_images", resolve)
    monkeypatch.setattr(ComposeBrokerClient, "resolve_images", resolve)
    return calls
