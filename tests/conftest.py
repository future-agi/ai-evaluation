"""Suite-wide fixtures.

The telemetry run ledger is always-on for real users; without an override the
test suite would append hundreds of rows to the developer's real ledger at
``~/.agent-learning/ledger/`` (and could leave a stale ``sync.cursor``).
Point every test at an ephemeral ledger directory instead. Tests that assert
ledger behaviour explicitly set ``AGENT_LEARNING_LEDGER_PATH`` themselves via
``monkeypatch``, which takes precedence over this session-scoped default.
"""

from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def _offline_telemetry_mode():
    """Phase 14: pin the W&B-style sync mode to ``local`` for the whole suite so
    NO test or gate subprocess makes a surprise network emit, even if the dev's
    env has FI keys exported (the P8 "stray key" concern). Set via ``os.environ``
    (not monkeypatch) so spawned lane subprocesses inherit it. Tests that exercise
    the keyed/auto path set ``AGENT_LEARNING_SYNC=auto`` themselves via monkeypatch.
    """

    prior = os.environ.get("AGENT_LEARNING_SYNC")
    if prior is None:
        os.environ["AGENT_LEARNING_SYNC"] = "local"
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("AGENT_LEARNING_SYNC", None)


@pytest.fixture(scope="session", autouse=True)
def _isolated_run_ledger():
    if os.environ.get("AGENT_LEARNING_LEDGER_PATH"):
        yield
        return
    # ignore_cleanup_errors: a real engine run writes ledger rows concurrently,
    # which can leave the dir "not empty" at teardown on some platforms — the
    # isolation goal (keep rows out of the dev's real ledger) is met regardless.
    with tempfile.TemporaryDirectory(
        prefix="agent-learning-test-ledger-", ignore_cleanup_errors=True
    ) as tmp:
        os.environ["AGENT_LEARNING_LEDGER_PATH"] = tmp
        try:
            yield
        finally:
            os.environ.pop("AGENT_LEARNING_LEDGER_PATH", None)
