"""A recorded mailbox greeting is chosen by style, and its own tone is never doubled."""

import json

from fi.alk.harness import voicemail_audio


def _catalog(tmp_path, entries):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def test_no_catalogue_means_no_clip(tmp_path, monkeypatch):
    monkeypatch.setenv(voicemail_audio.CATALOG_ENV, str(tmp_path / "absent.json"))
    assert voicemail_audio.clip_for("personal") is None


def test_a_clip_is_chosen_by_style(tmp_path, monkeypatch):
    audio = tmp_path / "greeting.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setenv(
        voicemail_audio.CATALOG_ENV,
        str(
            _catalog(
                tmp_path,
                [
                    {
                        "id": "A",
                        "style": "carrier",
                        "path": str(audio),
                        "has_tone": True,
                    },
                    {
                        "id": "B",
                        "style": "personal",
                        "path": str(audio),
                        "has_tone": False,
                    },
                ],
            )
        ),
    )
    carrier = voicemail_audio.clip_for("carrier")
    personal = voicemail_audio.clip_for("personal")
    assert carrier["id"] == "A" and carrier["has_tone"] is True
    assert personal["id"] == "B" and personal["has_tone"] is False


def test_a_style_the_catalogue_does_not_cover_falls_back(tmp_path, monkeypatch):
    audio = tmp_path / "greeting.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setenv(
        voicemail_audio.CATALOG_ENV,
        str(_catalog(tmp_path, [{"id": "A", "style": "carrier", "path": str(audio)}])),
    )
    assert voicemail_audio.clip_for("operator") is None
    assert voicemail_audio.clip_for("full") is None


def test_an_entry_whose_file_is_missing_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv(
        voicemail_audio.CATALOG_ENV,
        str(
            _catalog(
                tmp_path,
                [
                    {
                        "id": "gone",
                        "style": "personal",
                        "path": str(tmp_path / "nope.wav"),
                    },
                ],
            )
        ),
    )
    assert voicemail_audio.clip_for("personal") is None


def test_a_url_is_used_as_it_stands(tmp_path, monkeypatch):
    monkeypatch.setenv(
        voicemail_audio.CATALOG_ENV,
        str(
            _catalog(
                tmp_path,
                [
                    {
                        "id": "hosted",
                        "style": "personal",
                        "url": "https://example.test/vm.wav",
                    },
                ],
            )
        ),
    )
    assert (
        voicemail_audio.clip_for("personal")["source"] == "https://example.test/vm.wav"
    )


def test_an_unreadable_catalogue_is_not_an_error(tmp_path, monkeypatch):
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv(voicemail_audio.CATALOG_ENV, str(broken))
    assert voicemail_audio.clip_for("personal") is None


def test_the_shipped_catalogue_covers_the_styles_it_claims():
    """The supplied clips, read through the real resolver."""
    carrier = voicemail_audio.clip_for("carrier")
    personal = voicemail_audio.clip_for("personal")
    if carrier is None and personal is None:
        return  # the catalogue is local only, so its absence is legitimate
    assert carrier["has_tone"] is True, "the generic clip ends with its own tone"
    assert personal["has_tone"] is False, "the personal clips need a tone appended"


def test_the_clips_words_travel_with_it(tmp_path, monkeypatch):
    """The greeting is spoken as the mailbox's turn, so the words have to reach the call or the
    transcript shows the agent talking to nobody."""
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            [
                {
                    "id": "VM_X",
                    "style": "carrier",
                    "file_name": "greeting.wav",
                    "path": str(tmp_path / "greeting.wav"),
                    "has_tone": True,
                    "transcript": "No one is available to take your call.",
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "greeting.wav").write_bytes(b"RIFF")
    monkeypatch.setenv("ALK_VOICEMAIL_CATALOG", str(catalog))

    chosen = voicemail_audio.clip_for("carrier")
    assert chosen["transcript"] == "No one is available to take your call."
    assert chosen["has_tone"] is True


def test_a_clip_has_to_speak_the_scenarios_language(tmp_path, monkeypatch):
    """A mailbox greeting in the wrong language is worse than none: the session speaks its own
    greeting in the language the scenario asked for, and a recording cannot."""
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            [
                {
                    "id": "VM_EN",
                    "style": "carrier",
                    "language": "en",
                    "file_name": "greeting.wav",
                    "path": str(tmp_path / "greeting.wav"),
                    "has_tone": True,
                    "transcript": "The person you called is not available.",
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "greeting.wav").write_bytes(b"RIFF")
    monkeypatch.setenv("ALK_VOICEMAIL_CATALOG", str(catalog))

    assert voicemail_audio.clip_for("carrier", "en")["id"] == "VM_EN"
    # A regional tag still matches the language it belongs to.
    assert voicemail_audio.clip_for("carrier", "en-GB")["id"] == "VM_EN"
    # No language asked for means English, which is what the catalogue holds.
    assert voicemail_audio.clip_for("carrier")["id"] == "VM_EN"
    assert voicemail_audio.clip_for("carrier", "hi") is None
    assert voicemail_audio.clip_for("carrier", "es-MX") is None
