from __future__ import annotations

import uuid

import pytest

from fi.alk.episode import EPISODE_ID_NAMESPACE, derive_episode_event_id, derive_episode_id
from fi.simulate.runtime.ids import derive_event_id

TENANT = "0b7e6a52-3c1d-4f6e-9a61-2f1c5d8e9a10"
EPISODE_ID = "episode_d8fae3c7fd6153fcb22587195d661c2d"


def test_episode_id_vector():
    assert EPISODE_ID_NAMESPACE == uuid.UUID("79626429-64cd-4ec6-88e8-97b61bb649f3")
    assert derive_episode_id(TENANT, "vapi", "call-0001") == EPISODE_ID


def test_event_id_vectors():
    assert derive_episode_event_id(EPISODE_ID, "vapi", 0) == "event_a6830e999a7f5140b02683928040bdc7"
    assert derive_episode_event_id(EPISODE_ID, "vapi", 1) == "event_bb9da3b7b9f45c3c93ee4d9897baeb98"


@pytest.mark.parametrize("seq", [0, 1, 2, 17, 1000])
@pytest.mark.parametrize("source", ["vapi", "retell", "runtime"])
def test_event_id_equals_derive_event_id(seq, source):
    assert derive_episode_event_id(EPISODE_ID, source, seq) == derive_event_id(EPISODE_ID, source, seq)


def test_child_span_id_vector():
    # Ingestion's child-span formula, restated here because this package never
    # computes it: uuid5 of project, Episode id and the opening event's id.
    provider_span_namespace = uuid.UUID("4d61d4e2-7b3c-4a1e-9f02-2c6a5b8e1d70")
    project_id = "11111111-1111-4111-8111-111111111111"
    event_id = derive_episode_event_id(EPISODE_ID, "vapi", 1)
    child_span_id = uuid.uuid5(provider_span_namespace, f"{project_id}:{EPISODE_ID}:{event_id}").hex[:16]
    assert child_span_id == "c3996cf32d9153d7"
