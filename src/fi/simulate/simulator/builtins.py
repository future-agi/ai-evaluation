"""Built-in simulator-policy descriptors + registration.

The concrete synthetic-user behaviour is still owned by the chat/voice environment
loops today (the loop builds the persona-driven user). These descriptors give the
``simulator_registry`` a real entry per built-in ``simulator.adapter`` name so the
planner can *validate* the name (typo → clear error) and tools can enumerate what's
available. They carry a manifest, not a dispatchable policy — full registry dispatch
of the simulator is a separate, larger refactor.
"""

from __future__ import annotations

from dataclasses import dataclass

from fi.simulate.registry import register_simulator


@dataclass(frozen=True)
class SimulatorManifest:
    name: str
    modalities: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class SimulatorPolicyDescriptor:
    """A named, manifest-carrying handle for a built-in simulator policy."""

    manifest: SimulatorManifest


_SIMULATORS = [
    SimulatorPolicyDescriptor(
        SimulatorManifest(
            "synthetic_user",
            modalities=("text",),
            notes="LLM persona-driven synthetic user (chat/text loop).",
        )
    ),
    SimulatorPolicyDescriptor(
        SimulatorManifest(
            "livekit_simulator",
            modalities=("voice",),
            notes="LiveKit STT->LLM->TTS synthetic caller (voice loop).",
        )
    ),
]

for _descriptor in _SIMULATORS:
    register_simulator(_descriptor.manifest.name, _descriptor)


__all__ = ["SimulatorManifest", "SimulatorPolicyDescriptor"]
