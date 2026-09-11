"""Auto-skip for opt-in live lanes (PRD §4.1): collected, then SKIPPED
unless the lane's env flag is set. Release flows never set these flags.
Lives in tests/live/ so the existing tests are untouched by construction."""

import os

import pytest

# marker -> (env flag, extra) — the three facts every skip line names
LANE_MARKERS = {
    "live_livekit": ("AGENT_LEARNING_LIVE_LIVEKIT", "livekit"),
    "live_pipecat": ("AGENT_LEARNING_LIVE_PIPECAT", "pipecat"),
    "live_langchain": ("AGENT_LEARNING_LIVE_LANGCHAIN", "langchain"),
    "live_mcp": ("AGENT_LEARNING_LIVE_MCP", "mcp"),
    "live_a2a": ("AGENT_LEARNING_LIVE_A2A", "a2a"),
}


def _skip_reason(flag: str, extra: str) -> str:
    # The three-fact contract, asserted verbatim by a 3A test (guide §5.2):
    return (f"opt-in live lane: set {flag}=1 "
            f"(extra: {extra}; boundary: live_lane_boundary gate)")


def pytest_collection_modifyitems(config, items):
    any_lane_flag_set = any(
        os.environ.get(flag) == "1" for flag, _ in LANE_MARKERS.values()
    )
    for item in items:
        own = [(flag, extra) for marker, (flag, extra) in LANE_MARKERS.items()
               if marker in item.keywords]
        if "live_lane" in item.keywords:
            # umbrella rule (there is NO umbrella env flag): skip unless at
            # least one lane flag is set AND this test's specific lane flag
            # is set. A live_lane test without a specific lane marker is a
            # spec bug — it skips unconditionally and trips the 3A meta-test.
            enabled = (any_lane_flag_set and own
                       and all(os.environ.get(f) == "1" for f, _ in own))
            if not enabled:
                flag, extra = own[0] if own else ("<lane flag>", "<extra>")
                item.add_marker(pytest.mark.skip(reason=_skip_reason(flag, extra)))
        if ("live_credentialed" in item.keywords
                and os.environ.get("AGENT_LEARNING_LIVE_CREDENTIALED") != "1"):
            item.add_marker(pytest.mark.skip(
                reason="credentialed rung: set AGENT_LEARNING_LIVE_CREDENTIALED=1 + creds"))
