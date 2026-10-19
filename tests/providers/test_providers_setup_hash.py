from __future__ import annotations

import copy
import hashlib

from fi.alk.harness.outbound import canonical_bytes
from fi.alk.providers.setup_hash import compute_setup_hash


def test_formula_matches_canonical_bytes() -> None:
    agent = {
        "name": "Support line",
        "provider_ref": {"provider": "vapi", "id": "a1", "version": "7"},
        "direction_answer": "inbound",
        "setup_hash": "sh1:old",
        "extension": {"assistant": {"name": "Söderhamn"}},
    }
    before = copy.deepcopy(agent)
    hashed = {
        "name": "Support line",
        "provider_ref": {"provider": "vapi", "id": "a1"},
        "extension": {"assistant": {"name": "Söderhamn"}},
    }

    assert (
        compute_setup_hash(agent)
        == "sh1:" + hashlib.sha256(canonical_bytes(hashed)).hexdigest()
    )
    assert agent == before
