"""Persona & Scenario Studio — the Phase-7 common surface (ARCH §2a).

ONE studio package: re-exports the evolved ``fi.simulate`` engine classes
lazily (one class, one home — no parallel format) plus the studio API
(create / validate / calibrate / admit / lint / expand / coverage / import /
pull). The dependency points one way (facade -> ``fi.*``): class layers,
the behavior-policy compiler, realization metrics, and fidelity math live
ENGINE-side; this package never imports ``fi.alk.live`` (the
live_lane_boundary rule).
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    # evolved engine classes (lazy — the redteam.py _manifest()/_simulate() idiom)
    "Persona": ("fi.simulate.simulation.models", "Persona"),
    "Scenario": ("fi.simulate.simulation.models", "Scenario"),
    "PersonaIdentity": ("fi.simulate.simulation.models", "PersonaIdentity"),
    "PersonaTemperament": ("fi.simulate.simulation.models", "PersonaTemperament"),
    "BehaviorPolicy": ("fi.simulate.simulation.models", "BehaviorPolicy"),
    "PersonaFact": ("fi.simulate.simulation.models", "PersonaFact"),
    "AttackConditioning": ("fi.simulate.simulation.models", "AttackConditioning"),
    "PersonaProvenance": ("fi.simulate.simulation.models", "PersonaProvenance"),
    "EscalationArc": ("fi.simulate.simulation.models", "EscalationArc"),
    # Phase 13D simulation contract (one class, one home — ARCH §2.0 last row)
    "Simulation": ("fi.simulate.simulation.contract", "Simulation"),
    "ScenarioBinding": ("fi.simulate.simulation.contract", "ScenarioBinding"),
    "CastMember": ("fi.simulate.simulation.contract", "CastMember"),
    # engine fidelity facades
    "persona_fidelity": ("fi.simulate.simulation.fidelity", "persona_fidelity"),
    "attach_fidelity": ("fi.simulate.simulation.fidelity", "attach_fidelity"),
    # in-character fidelity as attack quality (unit 8)
    "attack_quality": ("fi.alk.studio._fidelity_attack", "attack_quality"),
    "persona_conditioned_campaign": (
        "fi.alk.studio._fidelity_attack",
        "persona_conditioned_campaign",
    ),
    # studio API
    "build_persona": ("fi.alk.studio._calibration", "build_persona"),
    "validate_persona": ("fi.alk.studio._calibration", "validate_persona"),
    "calibrate_persona": ("fi.alk.studio._calibration", "calibrate_persona"),
    "upgrade_legacy_persona": ("fi.alk.studio._upgrade", "upgrade_legacy_persona"),
    "expand_scenarios": ("fi.alk.studio._coverage", "expand_scenarios"),
    "synthesize_next_scenario": ("fi.alk.studio._coverage", "synthesize_next_scenario"),
    "coverage_report": ("fi.alk.studio._coverage", "coverage_report"),
    "residual_uncovered_estimate": ("fi.alk.studio._coverage", "residual_uncovered_estimate"),
    "bias_lint": ("fi.alk.studio._bias", "bias_lint"),
    "import_vendor_persona": ("fi.alk.studio._vendor", "import_vendor_persona"),
    "render_vendor_text": ("fi.alk.studio._vendor", "render_vendor_text"),
    "pull_personas": ("fi.alk.studio._download", "pull_personas"),
    "pull_scenarios": ("fi.alk.studio._download", "pull_scenarios"),
    "PlatformAgentReference": ("fi.alk.studio._generate", "PlatformAgentReference"),
    "PlatformScenarioRequest": ("fi.alk.studio._generate", "PlatformScenarioRequest"),
    "GeneratedScenario": ("fi.alk.studio._generate", "GeneratedScenario"),
    "ScenarioGenerationError": ("fi.alk.studio._generate", "ScenarioGenerationError"),
    "ensure_platform_agent": ("fi.alk.studio._generate", "ensure_platform_agent"),
    "generate_scenario": ("fi.alk.studio._generate", "generate_scenario"),
    "fetch_scenario": ("fi.alk.studio._generate", "fetch_scenario"),
    "load_persona": ("fi.alk.studio._library", "load_persona"),
    "save_persona": ("fi.alk.studio._library", "save_persona"),
    "load_scenario": ("fi.alk.studio._library", "load_scenario"),
    "save_scenario": ("fi.alk.studio._library", "save_scenario"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attribute = _EXPORTS[name]
        value = getattr(importlib.import_module(module_name), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


__all__ = sorted(_EXPORTS)
