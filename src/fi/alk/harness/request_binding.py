"""Request binding for release tool servers (`environment-releases.md` §6.2).

Every tool request a provider clone sends is bound to the scenario currently running on its world.
The runtime gives each world's clone a secret header; before each call or chat is created the
runner registers a fresh token for the scenario, which the provider echoes back on every tool
request. The tool server executes a request only when it presents both, so a late or retried
call from an earlier scenario can never change the world the next scenario is graded on.

World reset restores the template database, so these tables start empty for every scenario.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Mapping
from typing import Any

SECRET_HEADER = "x-alk-tool-secret"
BINDING_HEADER = "x-alk-binding"
BINDING_KEY = "alk_binding"
# Where the runtime keeps a world's clone secret (`ProcessRuntime` world metadata).
SECRET_METADATA_KEY = "provider_tool_secret"

ACTIVE_CALL_TABLE = "_alk_active_call"
TOOL_RESPONSES_TABLE = "_alk_tool_responses"
STRAY_CALLS_TABLE = "_alk_stray_calls"

# Appended to a release world's seed after the shared prefix, which creates `_alk_tool_trace`.
REQUEST_BINDING_SQL = """\
CREATE TABLE IF NOT EXISTS _alk_active_call (
  slot smallint PRIMARY KEY CHECK (slot = 1),
  token text NOT NULL,
  secret_sha256 text NOT NULL,
  scenario_key text NOT NULL,
  registered_at double precision NOT NULL
);
CREATE TABLE IF NOT EXISTS _alk_tool_responses (
  provider_call_id text NOT NULL,
  tool_call_id text NOT NULL,
  response jsonb NOT NULL,
  at double precision NOT NULL,
  PRIMARY KEY (provider_call_id, tool_call_id)
);
CREATE TABLE IF NOT EXISTS _alk_stray_calls (
  id bigserial PRIMARY KEY,
  provider_call_id text,
  token text,
  tool text,
  arguments jsonb,
  reason text NOT NULL,
  at double precision NOT NULL
);
ALTER TABLE _alk_tool_trace
  ADD COLUMN IF NOT EXISTS agent_tool text,
  ADD COLUMN IF NOT EXISTS provider_call_id text,
  ADD COLUMN IF NOT EXISTS tool_call_id text,
  ADD COLUMN IF NOT EXISTS status text;
"""


def new_secret() -> str:
    """One world's clone secret."""
    return secrets.token_urlsafe(32)


def new_token() -> str:
    """One call's binding token."""
    return secrets.token_urlsafe(16)


def secret_sha256(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def secret_matches(presented: str | None, expected_sha256: str) -> bool:
    if not presented or not expected_sha256:
        return False
    return hmac.compare_digest(secret_sha256(presented), expected_sha256)


def call_metadata(token: str) -> dict[str, str]:
    """What a runner attaches to the provider call or chat it creates (metadata and variables)."""
    return {BINDING_KEY: token}


def register_active_call(
    dsn: str, *, token: str, secret: str, scenario_key: str
) -> None:
    """Make `token` the only call this world's tool server executes, until the next one."""
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(
            f"INSERT INTO {ACTIVE_CALL_TABLE} "
            "(slot, token, secret_sha256, scenario_key, registered_at) "
            "VALUES (1, %s, %s, %s, %s) ON CONFLICT (slot) DO UPDATE SET "
            "token = EXCLUDED.token, secret_sha256 = EXCLUDED.secret_sha256, "
            "scenario_key = EXCLUDED.scenario_key, registered_at = EXCLUDED.registered_at",
            (token, secret_sha256(secret), scenario_key, time.time()),
        )


def read_stray_calls(dsn: str) -> list[dict[str, Any]]:
    """Requests the tool server refused to bind since the world was last reset."""
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as connection:
        cursor = connection.execute(
            "SELECT provider_call_id, token, tool, arguments, reason, at "
            f"FROM {STRAY_CALLS_TABLE} ORDER BY id"
        )
        columns = [column.name for column in cursor.description or []]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _at(value: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _header(headers: Mapping[str, str] | Any, name: str) -> str | None:
    getter = getattr(headers, "get", None)
    if getter is None:
        return None
    value = getter(name)
    if value is None and isinstance(headers, Mapping):
        value = next(
            (item for key, item in headers.items() if str(key).lower() == name), None
        )
    return str(value).strip() or None if value is not None else None


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def token_from_request(headers: Mapping[str, str] | Any, body: Any) -> str | None:
    """The binding token, from the first carrier present (contract §6.2)."""
    candidates = (
        _header(headers, BINDING_HEADER),
        _at(body, "call", "metadata", BINDING_KEY),
        _at(body, "chat", "metadata", BINDING_KEY),
        _at(body, "message", "call", "metadata", BINDING_KEY),
        _at(body, "call", "retell_llm_dynamic_variables", BINDING_KEY),
        _at(
            body,
            "message",
            "call",
            "assistantOverrides",
            "variableValues",
            BINDING_KEY,
        ),
    )
    return next((token for token in map(_text, candidates) if token), None)


def presented_secret(headers: Mapping[str, str] | Any) -> str | None:
    return _header(headers, SECRET_HEADER)


def provider_call_id(body: Any) -> str | None:
    """Vapi `message.call.id`, Retell `call.call_id`, or Retell chat `chat.chat_id`."""
    for path in (("message", "call", "id"), ("call", "call_id"), ("chat", "chat_id")):
        value = _text(_at(body, *path))
        if value:
            return value
    return None


def stray_arguments(arguments: Any) -> str:
    """Arguments as stored on a stray row: JSON, whatever the provider sent."""
    return json.dumps(arguments, default=str)
