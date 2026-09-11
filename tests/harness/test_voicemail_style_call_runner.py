"""The kind of mailbox travels to the caller's lane the same way the mailbox itself does."""

from __future__ import annotations

from pathlib import Path

from test_voicemail_call_runner import _drive


def test_the_style_reaches_the_environment(tmp_path: Path) -> None:
    environ = _drive(
        tmp_path,
        call_direction="outbound",
        answered_by="voicemail",
        voicemail_style="operator",
    )
    assert environ.get("HARNESS_ANSWERED_BY") == "voicemail"
    assert environ.get("HARNESS_VOICEMAIL_STYLE") == "operator"


def test_a_mailbox_with_no_style_named_leaves_it_unset(tmp_path: Path) -> None:
    """Unset is personal, decided where the greeting is written rather than here."""
    environ = _drive(tmp_path, call_direction="outbound", answered_by="voicemail")
    assert environ.get("HARNESS_ANSWERED_BY") == "voicemail"
    assert "HARNESS_VOICEMAIL_STYLE" not in environ


def test_a_person_answering_clears_a_style_left_by_the_previous_call(
    tmp_path: Path,
) -> None:
    """Scenarios share a process, so a stale style would give the next caller a mailbox greeting."""
    environ = _drive(tmp_path, call_direction="outbound", caller_awareness="expecting")
    assert "HARNESS_ANSWERED_BY" not in environ
    assert "HARNESS_VOICEMAIL_STYLE" not in environ


def test_an_inbound_call_clears_the_style_too(tmp_path: Path) -> None:
    environ = _drive(tmp_path, call_direction="inbound")
    assert "HARNESS_VOICEMAIL_STYLE" not in environ
