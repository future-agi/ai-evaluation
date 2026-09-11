from fi.alk.harness.backends.vertex_gemini import _successful_terminal_save


def test_successful_authoring_save_is_terminal() -> None:
    assert _successful_terminal_save(
        "mcp__world__save_world", {"content": "Saved to /work/authoring."}
    )
    assert _successful_terminal_save(
        "mcp__provision__save_environment", {"content": "Saved."}
    )
    assert _successful_terminal_save(
        "mcp__scenarios__save_scenarios", {"content": "Saved 10 scenarios."}
    )
    assert _successful_terminal_save(
        "mcp__source_data__finish_review", {"content": "Saved 12 invariants."}
    )


def test_rejected_save_and_non_save_tools_are_not_terminal() -> None:
    assert not _successful_terminal_save(
        "mcp__world__save_world", {"is_error": True, "content": "Not saved."}
    )
    assert not _successful_terminal_save(
        "mcp__world__check_world", {"content": "All checks pass."}
    )
    assert not _successful_terminal_save("mcp__world__save_world", "Saved")
