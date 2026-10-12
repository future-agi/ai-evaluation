"""Durable, replayable call-usage reporting for one hosted attempt."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .job import FailureDomain
from .outbound import HostedCapabilities, Transport, TransportError


logger = logging.getLogger(__name__)
USAGE_SCHEMA_VERSION = "futureagi.harness-usage.v1"
SIMULATOR_FUNDING_ALIAS = "ALK_SIMULATOR_FUNDING"
UsageAction = Literal["text_call", "voice_call"]
UsageOutcome = Literal["completed", "failed"]
Funding = Literal["platform", "customer"]
_FAILURE_DOMAINS_BY_CODE = {
    "usage_exhausted": FailureDomain.PLATFORM_SYNC,
    "usage_check_failed": FailureDomain.PLATFORM_SYNC,
    "world_unavailable": FailureDomain.ENVIRONMENT,
    "target_agent_stalled": FailureDomain.AGENT,
    "target_agent_tool_failed": FailureDomain.AGENT,
    "simulator_stalled": FailureDomain.SIMULATOR,
    "driver_crashed": FailureDomain.SIMULATOR,
    "evidence_missing": FailureDomain.SIMULATOR,
}


def failure_domain_for_code(code: str | None) -> FailureDomain:
    """Map the scheduler's closed failure codes to the usage-report domain."""

    return _FAILURE_DOMAINS_BY_CODE.get(
        str(code or ""),
        FailureDomain.INFRASTRUCTURE,
    )


class UsageUnavailable(RuntimeError):
    """A paid action could not be authorized because metering was unavailable."""


class UsageDenied(RuntimeError):
    """The platform rejected a paid action for insufficient usage allowance."""

    def __init__(self, detail: dict[str, Any] | None = None) -> None:
        self.detail = detail or {}
        nested = self.detail.get("result")
        effective = nested if isinstance(nested, dict) else self.detail
        super().__init__(
            str(
                effective.get("reason")
                or effective.get("message")
                or "usage limit reached"
            )
        )


class UsageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: uuid.UUID
    action: UsageAction
    scenario_key: str = Field(min_length=1)
    amount: float = Field(ge=0, allow_inf_nan=False)
    occurred_at: datetime
    funding: Funding
    outcome: UsageOutcome = "completed"
    failure_domain: FailureDomain | None = None

    @model_validator(mode="after")
    def _validate_failure(self) -> "UsageRecord":
        if self.outcome == "failed" and self.failure_domain is None:
            raise ValueError("failed usage records must include failure_domain")
        if self.outcome == "completed" and self.failure_domain is not None:
            raise ValueError("completed usage records cannot include failure_domain")
        return self


class UsageJournal:
    """An atomic cumulative snapshot whose immutable records are safe to replay."""

    def __init__(self, path: Path, *, attempt_id: str) -> None:
        self.path = path
        self.attempt_id = attempt_id
        self._lock = threading.RLock()
        self._records: list[UsageRecord] = []
        self._load()
        self._flush()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            raise UsageUnavailable(f"usage journal unreadable: {exc}") from exc
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != USAGE_SCHEMA_VERSION
        ):
            raise UsageUnavailable("usage journal schema is unsupported")
        try:
            self._records = [
                UsageRecord.model_validate(item) for item in raw.get("records", [])
            ]
        except (TypeError, ValueError) as exc:
            raise UsageUnavailable(f"usage journal record invalid: {exc}") from exc

    @property
    def records(self) -> tuple[UsageRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def append(
        self,
        *,
        action: UsageAction,
        scenario_key: str,
        amount: float,
        funding: Funding,
        occurred_at: datetime | None = None,
        outcome: UsageOutcome = "completed",
        failure_domain: FailureDomain | None = None,
        record_key: str | None = None,
    ) -> UsageRecord:
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("usage amount must be nonnegative and finite")
        with self._lock:
            if record_key is None:
                ordinal = 1 + sum(
                    record.action == action and record.scenario_key == scenario_key
                    for record in self._records
                )
                record_key = str(ordinal)
            record_id = uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"futureagi:harness-usage:{self.attempt_id}:{action}:{scenario_key}:{record_key}",
            )
            for existing in self._records:
                if existing.id != record_id:
                    continue
                if (
                    existing.action != action
                    or existing.scenario_key != scenario_key
                    or existing.amount != amount
                    or existing.funding != funding
                    or existing.outcome != outcome
                    or existing.failure_domain != failure_domain
                ):
                    raise UsageUnavailable(
                        f"usage record {record_id} was replayed with different facts"
                    )
                return existing
            record = UsageRecord(
                id=record_id,
                action=action,
                scenario_key=scenario_key,
                amount=amount,
                occurred_at=occurred_at or datetime.now(timezone.utc),
                funding=funding,
                outcome=outcome,
                failure_domain=failure_domain,
            )
            self._records.append(record)
            self._flush()
            return record

    def payload(self) -> dict[str, Any]:
        with self._lock:
            records = [record.model_dump(mode="json") for record in self._records]
        return {
            "schema_version": USAGE_SCHEMA_VERSION,
            "operation": "report",
            "records": records,
        }

    def _flush(self) -> None:
        payload = self.payload()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w",
            dir=self.path.parent,
            prefix=".usage-",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        )
        temporary = Path(handle.name)
        try:
            with handle:
                json.dump(
                    payload,
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)


def simulator_funding(environ: Mapping[str, str] | None = None) -> Funding:
    value = (environ or os.environ).get(SIMULATOR_FUNDING_ALIAS, "").strip().lower()
    return "customer" if value == "customer" else "platform"


def _usage_endpoint(scenarios_endpoint: str) -> str:
    parsed = urlsplit(scenarios_endpoint)
    parts = parsed.path.rstrip("/").split("/")
    if not parts or parts[-1] != "scenarios":
        raise UsageUnavailable("cannot derive usage endpoint from scenarios capability")
    parts[-1] = "usage"
    return urlunsplit(parsed._replace(path="/".join(parts) + "/"))


class UsageReporter:
    def __init__(
        self,
        capabilities: HostedCapabilities,
        transport: Transport,
        journal: UsageJournal,
    ) -> None:
        self._capabilities = capabilities
        self._transport = transport
        self.journal = journal
        self._endpoint = _usage_endpoint(capabilities.endpoints.scenarios)
        self._network_lock = threading.RLock()

    def check(self, action: UsageAction) -> None:
        last_transport_error: TransportError | None = None
        for attempt in range(3):
            try:
                with self._network_lock:
                    response = self._transport.request(
                        "POST",
                        self._endpoint,
                        headers=self._capabilities.auth_headers(),
                        json_body={"operation": "check", "action": action},
                    )
            except TransportError as exc:
                last_transport_error = exc
                if attempt < 2:
                    time.sleep(0.5 * (2**attempt))
                    continue
                raise UsageUnavailable(f"usage check unavailable: {exc}") from exc
            body = response.body
            result = (
                body.get("result")
                if isinstance(body, dict) and isinstance(body.get("result"), dict)
                else body
            )
            if response.status_code == 402:
                raise UsageDenied(body)
            if 500 <= response.status_code < 600 and attempt < 2:
                time.sleep(0.5 * (2**attempt))
                continue
            if response.status_code < 200 or response.status_code >= 300:
                raise UsageUnavailable(
                    f"usage check failed with HTTP {response.status_code}"
                )
            if not isinstance(result, dict) or result.get("allowed") is not True:
                raise UsageUnavailable("usage check response did not allow the action")
            return
        # The loop always returns or raises. Keep the invariant explicit for type checkers.
        raise UsageUnavailable(f"usage check unavailable: {last_transport_error}")

    def record(
        self,
        *,
        action: UsageAction,
        scenario_key: str,
        amount: float,
        funding: Funding,
        occurred_at: datetime | None = None,
        outcome: UsageOutcome = "completed",
        failure_domain: FailureDomain | None = None,
        record_key: str | None = None,
    ) -> UsageRecord:
        record = self.journal.append(
            record_key=record_key,
            action=action,
            scenario_key=scenario_key,
            amount=amount,
            funding=funding,
            occurred_at=occurred_at,
            outcome=outcome,
            failure_domain=failure_domain,
        )
        delivered = self.report()
        if not delivered:
            logger.error(
                "Hosted usage record journaled but not delivered for attempt %s: "
                "action=%s scenario=%s record_id=%s",
                self.journal.attempt_id,
                action,
                scenario_key,
                record.id,
            )
        return record

    def report(self) -> bool:
        try:
            with self._network_lock:
                response = self._transport.request(
                    "POST",
                    self._endpoint,
                    headers=self._capabilities.auth_headers(),
                    json_body=self.journal.payload(),
                )
        except TransportError as exc:
            logger.warning(
                "Hosted usage report transport failed for attempt %s: %s",
                self.journal.attempt_id,
                exc,
            )
            return False
        if not 200 <= response.status_code < 300:
            logger.warning(
                "Hosted usage report rejected for attempt %s with HTTP %s",
                self.journal.attempt_id,
                response.status_code,
            )
            return False
        return True
