"""C1 rejection-rule coverage for the widened TelephonyTransport.

Every ValueError promised by C1's "What the SDK promises" list gets a test
here, plus proof that widening the originator Literal and generalizing the
vapi validator strings left Vapi's behavior byte-identical.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fi.simulate.agent.definition import AgentDefinition


def _agent_def(**overrides):
    base = {"name": "a", "system_prompt": "p"}
    base.update(overrides)
    return AgentDefinition.model_validate(base)


def _reject(**kwargs) -> str:
    """Return the raw ValueError text (not pydantic's 'Value error, ...' wrapper)."""
    with pytest.raises(ValidationError) as exc:
        _agent_def(**kwargs)
    (err,) = exc.value.errors()
    return err["ctx"]["error"].args[0]


# --------------------------------------------------------------------------- #
# sip_outbound rejects originator fields
# --------------------------------------------------------------------------- #
def test_sip_outbound_rejects_inbound_call_originator() -> None:
    with pytest.raises(
        ValueError, match="sip_outbound cannot set inbound_call_originator"
    ):
        _agent_def(
            transport={
                "kind": "sip_outbound",
                "sip_trunk_id": "trunk_1",
                "sip_call_to": "+15551230000",
                "sip_number": "+15559990000",
                "inbound_call_originator": "retell",
            }
        )


def test_sip_outbound_rejects_originator_agent_id() -> None:
    # == (not match=) so an appended suffix on this C1-named string fails this test.
    assert (
        _reject(
            transport={
                "kind": "sip_outbound",
                "sip_trunk_id": "trunk_1",
                "sip_call_to": "+15551230000",
                "sip_number": "+15559990000",
                "originator_agent_id": "agent_1",
            }
        )
        == "sip_outbound cannot set originator fields"
    )


def test_sip_outbound_rejects_originator_from_number() -> None:
    # == (not match=) so an appended suffix on this C1-named string fails this test.
    assert (
        _reject(
            transport={
                "kind": "sip_outbound",
                "sip_trunk_id": "trunk_1",
                "sip_call_to": "+15551230000",
                "sip_number": "+15559990000",
                "originator_from_number": "+14155551234",
            }
        )
        == "sip_outbound cannot set originator fields"
    )


# --------------------------------------------------------------------------- #
# sip_inbound: originator fields require inbound_call_originator
# --------------------------------------------------------------------------- #
def test_sip_inbound_originator_agent_id_requires_originator() -> None:
    with pytest.raises(
        ValueError, match="originator fields require inbound_call_originator"
    ):
        _agent_def(transport={"kind": "sip_inbound", "originator_agent_id": "agent_1"})


def test_sip_inbound_originator_from_number_requires_originator() -> None:
    with pytest.raises(
        ValueError, match="originator fields require inbound_call_originator"
    ):
        _agent_def(
            transport={"kind": "sip_inbound", "originator_from_number": "+14155551234"}
        )


# --------------------------------------------------------------------------- #
# sip_inbound: the two originator fields are Retell-only (C1) — a Vapi
# originator reads its config from env and would otherwise silently ignore them.
# --------------------------------------------------------------------------- #
def test_sip_inbound_vapi_originator_rejects_originator_agent_id() -> None:
    assert (
        _reject(
            transport={
                "kind": "sip_inbound",
                "inbound_call_originator": "vapi",
                "originator_agent_id": "agent_1",
            }
        )
        == "vapi_originator_does_not_take_originator_fields"
    )


def test_sip_inbound_vapi_originator_rejects_originator_from_number() -> None:
    assert (
        _reject(
            transport={
                "kind": "sip_inbound",
                "inbound_call_originator": "vapi",
                "originator_from_number": "+14155551234",
            }
        )
        == "vapi_originator_does_not_take_originator_fields"
    )


def test_sip_inbound_retell_originator_accepts_both_originator_fields() -> None:
    agent = _agent_def(
        transport={
            "kind": "sip_inbound",
            "inbound_call_originator": "retell",
            "originator_agent_id": "agent_1",
            "originator_from_number": "+14155551234",
        },
        provider_evidence={
            "provider": "retell",
            "call_id_source": "originator_response",
        },
    )
    assert agent.transport.originator_agent_id == "agent_1"
    assert agent.transport.originator_from_number == "+14155551234"


# --------------------------------------------------------------------------- #
# sip_inbound: originator_from_number must be E.164
# --------------------------------------------------------------------------- #
def test_sip_inbound_rejects_non_e164_originator_from_number() -> None:
    with pytest.raises(ValueError, match=r"originator_from_number must be E\.164"):
        _agent_def(
            transport={
                "kind": "sip_inbound",
                "inbound_call_originator": "retell",
                "originator_from_number": "4155551234",
            }
        )


# --------------------------------------------------------------------------- #
# E.164 boundary pin — guards the module-level _E164 regex against drift
# (e.g. a `.startswith("+")` swap or a `^\+\d+$` loosening).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("number", ["+1234567", "+123456789012345"])
def test_originator_from_number_accepts_e164_boundaries(number: str) -> None:
    agent = _agent_def(
        transport={
            "kind": "sip_inbound",
            "inbound_call_originator": "retell",
            "originator_from_number": number,
        },
        provider_evidence={
            "provider": "retell",
            "call_id_source": "originator_response",
        },
    )
    assert agent.transport.originator_from_number == number


@pytest.mark.parametrize(
    "number",
    [
        "+123456",
        "+1234567890123456",
        "+01234567",
        "4155550123",
        "+1 415 555 0123",
        "+1415555012a",
        "+14155550123\n",
    ],
)
def test_originator_from_number_rejects_non_e164(number: str) -> None:
    with pytest.raises(ValueError, match=r"originator_from_number must be E\.164"):
        _agent_def(
            transport={
                "kind": "sip_inbound",
                "inbound_call_originator": "retell",
                "originator_from_number": number,
            }
        )


# --------------------------------------------------------------------------- #
# _E164 uses .fullmatch (not .match): "$" also matches before a trailing
# newline, so pin the two pre-existing sip_outbound sites against it too.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field", ["sip_call_to", "sip_number"])
def test_sip_outbound_rejects_trailing_newline(field: str) -> None:
    transport = {
        "kind": "sip_outbound",
        "sip_trunk_id": "trunk_1",
        "sip_call_to": "+15551230000",
        "sip_number": "+15559990000",
        field: "+14155551234\n",
    }
    with pytest.raises(ValueError, match=rf"sip_outbound requires E\.164 {field}"):
        _agent_def(transport=transport)


# --------------------------------------------------------------------------- #
# sip_inbound: originator_agent_id must be non-empty when set (None stays
# forward-compatible with jobs written before these fields existed).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("agent_id", ["", "   "])
def test_sip_inbound_rejects_empty_originator_agent_id(agent_id: str) -> None:
    # == (not match=) so an appended suffix on this C1-named string fails this test.
    assert (
        _reject(
            transport={
                "kind": "sip_inbound",
                "inbound_call_originator": "retell",
                "originator_agent_id": agent_id,
            }
        )
        == "originator_agent_id must be non-empty when set"
    )


def test_sip_inbound_accepts_none_originator_agent_id() -> None:
    agent = _agent_def(
        transport={
            "kind": "sip_inbound",
            "inbound_call_originator": "retell",
        },
        provider_evidence={
            "provider": "retell",
            "call_id_source": "originator_response",
        },
    )
    assert agent.transport.originator_agent_id is None


# --------------------------------------------------------------------------- #
# web transports: new fields join the existing any([...]) list
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["webrtc", "vapi_websocket", "retell_webcall"])
def test_web_transport_rejects_originator_agent_id(kind: str) -> None:
    with pytest.raises(ValueError, match=f"{kind} transport cannot set SIP fields"):
        _agent_def(transport={"kind": kind, "originator_agent_id": "agent_1"})


@pytest.mark.parametrize("kind", ["webrtc", "vapi_websocket", "retell_webcall"])
def test_web_transport_rejects_originator_from_number(kind: str) -> None:
    with pytest.raises(ValueError, match=f"{kind} transport cannot set SIP fields"):
        _agent_def(transport={"kind": kind, "originator_from_number": "+14155551234"})


# --------------------------------------------------------------------------- #
# evidence provider must equal the originator; call_id_source must be
# originator_response — Retell, and Vapi for byte-identical proof.
# --------------------------------------------------------------------------- #
def test_retell_originator_requires_retell_evidence() -> None:
    with pytest.raises(ValueError, match="retell_originator_requires_retell_evidence"):
        _agent_def(
            transport={
                "kind": "sip_inbound",
                "inbound_call_originator": "retell",
                "readiness_timeout_seconds": 120,
            },
            provider_evidence={
                "provider": "vapi",
                "call_id_source": "originator_response",
            },
        )


def test_retell_originator_requires_originator_response() -> None:
    with pytest.raises(
        ValueError, match="retell_originator_requires_originator_response"
    ):
        _agent_def(
            transport={
                "kind": "sip_inbound",
                "inbound_call_originator": "retell",
                "readiness_timeout_seconds": 120,
            },
            provider_evidence={
                "provider": "retell",
                "call_id_source": "participant_attribute",
                "participant_attribute": "sip.callID",
            },
        )


def test_vapi_originator_requires_vapi_evidence_byte_identical() -> None:
    # == (not match=) so an appended suffix on the templated string fails this test.
    assert (
        _reject(
            transport={
                "kind": "sip_inbound",
                "inbound_call_originator": "vapi",
                "readiness_timeout_seconds": 120,
            },
            provider_evidence={
                "provider": "retell",
                "call_id_source": "originator_response",
            },
        )
        == "vapi_originator_requires_vapi_evidence"
    )


def test_vapi_originator_requires_originator_response_byte_identical() -> None:
    # == (not match=) so an appended suffix on the templated string fails this test.
    assert (
        _reject(
            transport={
                "kind": "sip_inbound",
                "inbound_call_originator": "vapi",
                "readiness_timeout_seconds": 120,
            },
            provider_evidence={
                "provider": "vapi",
                "call_id_source": "participant_attribute",
                "participant_attribute": "sip.callID",
            },
        )
        == "vapi_originator_requires_originator_response"
    )


# --------------------------------------------------------------------------- #
# forward compatibility: a sip_inbound+retell job without the two new fields
# still validates (an SDK release without backend support must not break).
# --------------------------------------------------------------------------- #
def test_sip_inbound_retell_without_new_fields_still_validates() -> None:
    agent = _agent_def(
        transport={
            "kind": "sip_inbound",
            "inbound_call_originator": "retell",
            "readiness_timeout_seconds": 120,
        },
        provider_evidence={
            "provider": "retell",
            "call_id_source": "originator_response",
        },
    )
    assert agent.transport.originator_agent_id is None
    assert agent.transport.originator_from_number is None


def test_sip_inbound_retell_with_new_fields_validates() -> None:
    agent = _agent_def(
        transport={
            "kind": "sip_inbound",
            "inbound_call_originator": "retell",
            "originator_agent_id": "agent_123",
            "originator_from_number": "+14155550123",
            "readiness_timeout_seconds": 120,
        },
        provider_evidence={
            "provider": "retell",
            "call_id_source": "originator_response",
        },
    )
    assert agent.transport.originator_agent_id == "agent_123"
    assert agent.transport.originator_from_number == "+14155550123"


# --------------------------------------------------------------------------- #
# target + SIP transport is still rejected (C1 Forbidden list)
# --------------------------------------------------------------------------- #
def test_target_with_sip_transport_still_rejected() -> None:
    with pytest.raises(ValueError, match="retell_target_requires_retell_webcall"):
        _agent_def(
            transport={"kind": "sip_inbound", "readiness_timeout_seconds": 120},
            target={"provider": "retell", "agent_id": "agent_healthcare"},
        )


# --------------------------------------------------------------------------- #
# the Vapi originator fixture at oss/simulation-acceptance/voice_cases.py:862
# (case_id "2.2.1") must keep validating unchanged.
# --------------------------------------------------------------------------- #
def test_voice_cases_vapi_originator_fixture_still_validates() -> None:
    agent = AgentDefinition(
        name="vapi-originating-target",
        system_prompt="Copy the current target-agent prompt here.",
        transport={
            "kind": "sip_inbound",
            "inbound_call_originator": "vapi",
            "readiness_timeout_seconds": 120,
        },
        provider_evidence={
            "provider": "vapi",
            "call_id_source": "originator_response",
            "poll_deadline_seconds": 90,
        },
    )
    assert agent.transport.inbound_call_originator == "vapi"
    assert agent.transport.originator_agent_id is None
    assert agent.transport.originator_from_number is None
