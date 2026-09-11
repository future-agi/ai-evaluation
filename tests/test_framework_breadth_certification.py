"""Battle-test: every framework adapter preset certifies credential-free.

Durable regression guard for the "test the library standalone for all the
frameworks" directive: all 54 adapter presets must expose a valid, import-free
adapter contract (a method/input_mode the generic wrapper can drive). This is the
CONTRACT layer — running each on a REAL framework instance needs that framework
installed (owner-env); the adapter is import-free by design (it wraps any object
with the declared method), proven live on callable/litellm/openai/langchain.
"""

from __future__ import annotations

from fi.simulate.agent.frameworks import (
    FRAMEWORK_PRESETS,
    framework_adapter_contract,
)


def test_all_framework_presets_have_a_valid_contract() -> None:
    failures: list[tuple[str, str]] = []
    for framework in sorted(FRAMEWORK_PRESETS):
        try:
            contract = framework_adapter_contract(framework)
        except Exception as exc:  # noqa: BLE001
            failures.append((framework, f"{type(exc).__name__}: {exc}"))
            continue
        # a drivable contract declares an input_mode (and usually a method);
        # 'callable'/'custom' may carry method=None with a generic input_mode.
        if not contract.get("input_mode"):
            failures.append((framework, f"no input_mode: {contract!r}"))
    assert not failures, f"adapter contract failures: {failures}"


def test_framework_preset_count_is_the_full_breadth() -> None:
    # guards against a silent shrink of the supported-framework surface.
    assert len(FRAMEWORK_PRESETS) >= 54, len(FRAMEWORK_PRESETS)


def test_contract_carries_method_and_modality_fields() -> None:
    # spot-check the contract shape the generic wrapper relies on.
    for framework in ("langchain", "openai", "litellm", "crewai", "llamaindex"):
        contract = framework_adapter_contract(framework)
        assert "method" in contract
        assert contract.get("input_mode")
        assert "framework" in contract
