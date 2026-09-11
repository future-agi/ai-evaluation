"""Where an agent comes from, and how a session reaches it.

A folder of source code is one kind of agent, not the only kind. The same agent may arrive as a
provider connection with a system prompt and a tool schema, as a platform definition, or as a
spec somebody pasted in. The stage that reads an agent is the same in all of those cases; what
differs is where it looks and what it is allowed to touch.

So the method stays in the skill and the location lives here. Supporting a new kind of agent is
registering one class, not editing any stage.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class AgentSource(Protocol):
    """Everything a stage needs in order to reach one agent."""

    kind: str
    name: str

    def workdir(self) -> Path:
        """The directory the session runs in."""

    def builtin_tools(self) -> tuple[str, ...]:
        """Built-in tools this source needs granted."""

    def servers(self) -> dict[str, Any]:
        """In-process tool servers this source provides, if any."""

    def briefing(self) -> str:
        """What to tell the model about where this agent's truth lives."""


@dataclass
class RepoSource:
    """An agent that exists as source code on disk."""

    name: str
    root: Path
    kind: str = "repo"

    def workdir(self) -> Path:
        return self.root

    def builtin_tools(self) -> tuple[str, ...]:
        return ("Read", "Glob", "Grep")

    def servers(self) -> dict[str, Any]:
        return {}

    def briefing(self) -> str:
        ignored = {
            ".git",
            ".venv",
            "node_modules",
            "__pycache__",
            "artifacts",
            "build",
            "dist",
        }
        indexed: list[str] = []
        for path in sorted(self.root.rglob("*")):
            try:
                relative = path.relative_to(self.root)
            except ValueError:
                continue
            if any(part in ignored for part in relative.parts) or not path.is_file():
                continue
            indexed.append(relative.as_posix())
            if len(indexed) >= 240:
                break
        return (
            f"This agent is a repository at {self.root}. Its truth is the source code: the tool "
            "registrations, the function signatures, the validation logic, and whatever holds "
            "its data. Read it with Read, Glob and Grep. Documentation describes intent; the "
            "code describes behaviour, and where they disagree the code wins. Never glob the "
            "entire repository: dependency caches such as .venv and node_modules are irrelevant. "
            "Start from this pre-indexed source/config file list and read only relevant files:\n"
            + "\n".join(f"- {name}" for name in indexed)
        )


@dataclass
class GitHubSource(RepoSource):
    """A public GitHub repository cloned into this harness session."""

    url: str = ""
    kind: str = "github"

    def briefing(self) -> str:
        return (
            f"This agent was cloned from {self.url or 'GitHub'} into {self.root}. Its truth is "
            "the cloned source code: the tool registrations, function signatures, validation "
            "logic, and whatever holds its data. Read it with Read, Glob and Grep."
        )


def clone_github_repository(url: str, destination: Path) -> Path:
    """Shallow-clone one public GitHub repository into a session-owned directory."""
    from .github import parse_github_location

    try:
        location = parse_github_location(url)
    except ValueError as exc:
        raise ValueError(
            "use a public HTTPS GitHub repository or branch URL, such as "
            "https://github.com/owner/repo/tree/branch"
        ) from exc
    if destination.exists():
        raise ValueError(f"the session source directory already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["git", "clone", "--depth", "1"]
    if location.ref:
        command.extend(["--branch", location.ref])
    command.extend([location.clone_url, str(destination)])
    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or "git clone failed"
        raise RuntimeError(detail)
    return destination


@dataclass
class SpecSource:
    """An agent supplied directly as a prompt and a tool schema, with no repository.

    This is the shape a hosted provider gives back, so it is also the fallback whenever a
    connection can be read once and handed over as text.
    """

    name: str
    system_prompt: str
    tool_schema: list[dict[str, Any]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    scratch: Path = Path(".")
    kind: str = "spec"

    def workdir(self) -> Path:
        return self.scratch

    def builtin_tools(self) -> tuple[str, ...]:
        return ()

    def servers(self) -> dict[str, Any]:
        return {}

    def briefing(self) -> str:
        parts = [
            "This agent is supplied as a definition, not a repository. Everything knowable "
            "about it is below; there is no code to open, so do not guess at anything absent.",
            f"SYSTEM PROMPT:\n{self.system_prompt}",
        ]
        if self.tool_schema:
            parts.append(
                f"TOOL SCHEMA:\n{json.dumps(self.tool_schema, indent=2)[:6000]}"
            )
        if self.data:
            parts.append(f"DATA:\n{json.dumps(self.data, indent=2)[:6000]}")
        return "\n\n".join(parts)


@dataclass
class ProviderSource:
    """A sanitized definition fetched from an externally hosted provider.

    A connect-only provider agent has no repository in the sandbox.  Representing its empty
    source directory as a :class:`RepoSource` gives an authoring model filesystem tools and can
    make it wander outside that directory looking for an implementation.  The provider profile
    is the complete source of truth for this mode, so expose only that profile and no file tools.
    """

    name: str
    profile: dict[str, Any]
    scratch: Path = Path(".")
    kind: str = "provider"

    def workdir(self) -> Path:
        return self.scratch

    def builtin_tools(self) -> tuple[str, ...]:
        return ()

    def servers(self) -> dict[str, Any]:
        return {}

    def briefing(self) -> str:
        return (
            "This is an externally hosted provider agent, not a repository. The sanitized "
            "provider definition below is authoritative for its conversation, prompt, model, "
            "voice, states, and tool schemas. There is no source code to search or open. Do not "
            "invent behavior or tool inputs that are absent from this definition.\n\n"
            f"PROVIDER DEFINITION:\n{json.dumps(self.profile, indent=2, sort_keys=True)}"
        )


_REGISTRY: dict[str, Callable[..., AgentSource]] = {
    "repo": lambda **kw: RepoSource(name=kw["name"], root=Path(kw["root"])),
    "github": lambda **kw: GitHubSource(
        name=kw["name"], root=Path(kw["root"]), url=kw.get("url", "")
    ),
    "spec": lambda **kw: SpecSource(
        name=kw["name"],
        system_prompt=kw.get("system_prompt", ""),
        tool_schema=kw.get("tool_schema") or [],
        data=kw.get("data") or {},
        scratch=Path(kw.get("scratch", ".")),
    ),
    "provider": lambda **kw: ProviderSource(
        name=kw["name"],
        profile=kw.get("profile") or {},
        scratch=Path(kw.get("scratch", ".")),
    ),
}


def register_source(kind: str, factory: Callable[..., AgentSource]) -> None:
    """Add a kind of agent. A provider connection is a class and one line here."""
    _REGISTRY[kind] = factory


def resolve(kind: str, **kwargs: Any) -> AgentSource:
    if kind not in _REGISTRY:
        raise NotImplementedError(
            f"no agent source of kind {kind!r}; registered kinds are "
            f"{', '.join(sorted(_REGISTRY))}"
        )
    # An empty root used to resolve to the current directory, which is worse than failing: every
    # later stage then reads a real path, finds the harness's own repository, and reports that the
    # agent has no code on disk. Nothing downstream can tell that apart from an agent that really
    # was given as a specification.
    if "root" in kwargs and not str(kwargs.get("root") or "").strip():
        raise ValueError(
            f"a {kind!r} source needs the path its code lives at, and none was given. If this "
            "agent has no code on disk, it is not this kind of source."
        )
    return _REGISTRY[kind](**kwargs)


def supported() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
