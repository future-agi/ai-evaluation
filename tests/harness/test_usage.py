from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
import logging


import pytest

from fi.alk.harness.outbound import (
    CAPABILITIES_SCHEMA_VERSION,
    HostedCapabilities,
    HostedEndpoints,
    TransportError,
    TransportResponse,
)
from fi.alk.harness.usage import UsageJournal, UsageReporter, UsageUnavailable


def _capabilities() -> HostedCapabilities:
    root = "https://platform.example/simulate/api/harness-attempts/attempt-1"
    return HostedCapabilities(
        schema_version=CAPABILITIES_SCHEMA_VERSION,
        job_id="job-1",
        attempt_id="attempt-1",
        attempt_number=1,
        fence="fence-1",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        token="token-1",
        endpoints=HostedEndpoints(
            events=f"{root}/events/",
            results=f"{root}/results/",
            artifacts=f"{root}/artifacts/",
            scenarios=f"{root}/scenarios/",
        ),
    )


def test_usage_journal_recovers_and_deduplicates_stable_call(tmp_path) -> None:
    path = tmp_path / "usage.json"
    first = UsageJournal(path, attempt_id="attempt-1")
    original = first.append(
        action="voice_call",
        scenario_key="account-help",
        amount=1.25,
        funding="platform",
        record_key="1",
    )

    recovered = UsageJournal(path, attempt_id="attempt-1")
    replay = recovered.append(
        action="voice_call",
        scenario_key="account-help",
        amount=1.25,
        funding="platform",
        record_key="1",
    )

    assert replay.id == original.id
    assert len(recovered.records) == 1
    payload = json.loads(path.read_text())
    assert payload == {
        "operation": "report",
        "records": [original.model_dump(mode="json")],
        "schema_version": "futureagi.harness-usage.v1",
    }


def test_replayed_call_cannot_change_measured_amount(tmp_path) -> None:
    journal = UsageJournal(tmp_path / "usage.json", attempt_id="attempt-1")
    journal.append(
        action="text_call",
        scenario_key="account-help",
        amount=321,
        funding="platform",
        record_key="1",
    )

    with pytest.raises(UsageUnavailable, match="different facts"):
        journal.append(
            action="text_call",
            scenario_key="account-help",
            amount=322,
            funding="platform",
            record_key="1",
        )


class _UnavailableTransport:
    def request(self, *args, **kwargs):
        raise TransportError("offline")


class _RecordingTransport:
    def __init__(self) -> None:
        self.body = None

    def request(self, *args, **kwargs):
        self.body = kwargs["json_body"]
        return TransportResponse(200, {"result": {"allowed": True}}, {})


class _TransientTransport:
    def __init__(self) -> None:
        self.calls = 0

    def request(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise TransportError("temporary TLS failure")
        if self.calls == 2:
            return TransportResponse(503, {"detail": "busy"}, {})
        return TransportResponse(200, {"result": {"allowed": True}}, {})


def test_paid_usage_check_fails_closed_on_transport_error(tmp_path) -> None:
    reporter = UsageReporter(
        _capabilities(),
        _UnavailableTransport(),
        UsageJournal(tmp_path / "usage.json", attempt_id="attempt-1"),
    )

    with pytest.raises(UsageUnavailable, match="usage check unavailable"):
        reporter.check("voice_call")


def test_usage_record_logs_when_report_is_not_delivered(tmp_path, caplog) -> None:
    reporter = UsageReporter(
        _capabilities(),
        _UnavailableTransport(),
        UsageJournal(tmp_path / "usage.json", attempt_id="attempt-1"),
    )
    caplog.set_level(logging.ERROR)

    record = reporter.record(
        action="voice_call",
        scenario_key="account-help",
        amount=1.25,
        funding="platform",
    )

    assert record in reporter.journal.records
    assert "journaled but not delivered" in caplog.text
    assert str(record.id) in caplog.text


def test_usage_check_sends_only_the_call_action(tmp_path) -> None:
    transport = _RecordingTransport()
    reporter = UsageReporter(
        _capabilities(),
        transport,
        UsageJournal(tmp_path / "usage.json", attempt_id="attempt-1"),
    )

    reporter.check("text_call")

    assert transport.body == {"operation": "check", "action": "text_call"}


def test_usage_check_retries_transient_transport_and_server_failures(
    tmp_path, monkeypatch
) -> None:
    transport = _TransientTransport()
    reporter = UsageReporter(
        _capabilities(),
        transport,
        UsageJournal(tmp_path / "usage.json", attempt_id="attempt-1"),
    )
    monkeypatch.setattr("fi.alk.harness.usage.time.sleep", lambda _seconds: None)

    reporter.check("voice_call")

    assert transport.calls == 3
