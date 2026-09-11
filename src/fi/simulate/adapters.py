"""Named constants for the built-in adapter strings used across a ``SimulationSpec``.

These are **ergonomic sugar**, not a closed vocabulary. Every adapter slot on the
spec stays a plain ``str`` so third-party plugins registered by name (decorator or
``entry_points``) keep working. Each enum subclasses ``str``, so a member is the
string — ``EnvironmentSpec(adapter=EnvironmentAdapters.CHAT)`` is identical to
``adapter="chat"`` (same value, same ``spec_hash``). Use them for autocomplete,
typo-safety, and discoverability; drop to a raw string for anything custom.

``(str, Enum)`` rather than ``enum.StrEnum`` because the SDK floor is Python 3.10.
"""

from __future__ import annotations

from enum import Enum


class EnvironmentAdapters(str, Enum):
    """Built-in ``environment.adapter`` values — the **interaction loop** (how turns
    happen), not the world's contents. Tool mocking / stateful tools are a property
    of the *world object* (an ``EnvironmentAdapter``) that any loop can drive, not a
    loop of their own."""

    CHAT = "chat"
    VOICE = "voice"


class TargetAdapters(str, Enum):
    """Built-in ``target.adapter`` values (the agent-under-test / ActorSource)."""

    SYSTEM_PROMPT = "system_prompt"
    FACTORY = "factory"
    IMPORT_OBJECT = "import_object"
    FRAMEWORK = "framework"
    CALLABLE = "callable"
    PYTHON_CALLABLE = "python_callable"
    HTTP = "http"
    WEBSOCKET = "websocket"
    WEBRTC = "webrtc"
    LIVEKIT = "livekit"
    VAPI_WEBSOCKET = "vapi_websocket"
    RETELL_WEBCALL = "retell_webcall"
    SIP_INBOUND = "sip_inbound"
    SIP_OUTBOUND = "sip_outbound"


class SimulatorAdapters(str, Enum):
    """Built-in ``simulator.adapter`` values (the synthetic-user policy)."""

    SYNTHETIC_USER = "synthetic_user"
    LIVEKIT_SIMULATOR = "livekit_simulator"


class WorldKinds(str, Enum):
    """Built-in ``environment.world_kind`` labels — a **faithful mirror** of
    ``fi.simulate.simulation.contract.SIMULATION_WORLD_KINDS`` (the frozen
    single-home; ``test_worldkinds_mirror_contract`` byte-compares the two).

    A ``world_kind`` names the **primary surface / modality** of an episode —
    what is being exercised — not the engine and not "can a tool be called". It
    is an *admission + benchmark label*, never an executor selector: executable
    kinds (``conversation``, ``tool_api``) run contract-native rung-1 on the
    shared text loop; ``browser`` / ``voice_telephony`` run derived-legacy;
    ``computer_use`` / ``code_exec`` refuse (typed-only, engine staged).

    Tool *use* is orthogonal — a capability (``contract.ToolBinding`` +
    ``TOOL_MOCK_LEVELS``) attachable to ANY kind via ``WorldSpec.tools``; a
    ``conversation`` world may call tools too. ``tool_api`` is the kind where the
    tool surface *is* the world (scored as the tool-use modality), not "a world
    that happens to call tools". Legacy runtime aliases ``"chat"`` / ``"text"``
    (folded into ``conversation``) and ``"voice"`` (canonical:
    ``voice_telephony``) remain valid as raw strings; the enum carries canon."""

    CONVERSATION = "conversation"
    TOOL_API = "tool_api"
    BROWSER = "browser"
    COMPUTER_USE = "computer_use"
    CODE_EXEC = "code_exec"
    VOICE_TELEPHONY = "voice_telephony"


__all__ = [
    "EnvironmentAdapters",
    "TargetAdapters",
    "SimulatorAdapters",
    "WorldKinds",
]
