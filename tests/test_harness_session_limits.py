from fi.alk.harness.session import DEFAULT_STAGE_IDLE_TIMEOUT_SECONDS


def test_default_stage_idle_timeout_allows_long_parallel_generation() -> None:
    assert DEFAULT_STAGE_IDLE_TIMEOUT_SECONDS == 600.0
