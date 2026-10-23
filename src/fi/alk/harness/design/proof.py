"""The PostgreSQL proof over a design view while it is still being written.

A scenario whose solution calls an attached MCP server cannot be proved on SQLite, where those
servers do not run, so it is proved here when a writer submits it, the way the seal will prove
it. The proof seeds a template database from the current release view (reader grants included)
and keeps it while the view is unchanged; each attempt clones it. One proof runs at a time.
"""

from __future__ import annotations

import hashlib
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..releases import (
    ATTACHED,
    POLICY,
    SERVERS,
    TOOLS,
    PostgresProof,
    ReleaseError,
    ToolDefinition,
    postgres_verdict,
)
from ..world.snapshot import DATABASE, HANDLERS, MANIFEST, WORLD_MODULE, saved
from .workspace import load_design, release_view


class ViewProof:
    """The design view's PostgreSQL proof, reseeded only when the view changes."""

    def __init__(self, workspace: Path, admin_dsn: Callable[[], str]) -> None:
        self.workspace = Path(workspace)
        self._admin_dsn = admin_dsn
        self._lock = threading.Lock()
        self._digest = ""
        self._proof: PostgresProof | None = None
        self._scratch: tempfile.TemporaryDirectory[str] | None = None

    def _view_digest(self) -> str:
        digest = hashlib.sha256(
            load_design(self.workspace).policy_profile.model_dump_json().encode()
        )
        files = [
            self.workspace / name
            for name in (TOOLS, POLICY, SERVERS, MANIFEST, DATABASE)
        ]
        for folder in (ATTACHED, HANDLERS):
            files += sorted(
                path for path in (self.workspace / folder).rglob("*") if path.is_file()
            )
        for path in files:
            digest.update(str(path.relative_to(self.workspace)).encode() + b"\0")
            digest.update(path.read_bytes() if path.is_file() else b"\0missing")
        return digest.hexdigest()

    def _current(self) -> PostgresProof:
        if not saved(self.workspace) or not (self.workspace / WORLD_MODULE).is_file():
            raise ReleaseError(
                "design_world_missing",
                "the world has nothing in it yet: build it before anything is proved against it",
            )
        digest = self._view_digest()
        if self._proof is not None and digest == self._digest:
            return self._proof
        self._close()
        scratch = tempfile.TemporaryDirectory(prefix="design-proof-")
        try:
            environment, suite = release_view(
                self.workspace, Path(scratch.name) / "view"
            )
            # A world can attach a server before it defines a tool or a rule of its own.
            for name, empty in ((TOOLS, "[]"), (POLICY, '{"rules": []}')):
                if not (environment / name).is_file():
                    (environment / name).write_text(empty + "\n", encoding="utf-8")
            proof = PostgresProof(
                suite,
                environment,
                load_design(self.workspace).policy_profile,
                self._admin_dsn(),
            ).__enter__()
        except BaseException:
            scratch.cleanup()
            raise
        self._proof, self._scratch, self._digest = proof, scratch, digest
        return proof

    def scenario(self, scenario: Any, catalogue: Any) -> str:
        """Why `scenario` does not hold on PostgreSQL, graded with `catalogue`; "" when it does."""
        with self._lock:
            try:
                return postgres_verdict(self._current(), scenario, catalogue=catalogue)[
                    1
                ]
            except ReleaseError as exc:
                return f"{exc.code}: {exc.detail}"

    def definitions(self) -> list[ToolDefinition]:
        """The attached tools a world of this view offers, from each server's own `tools/list`."""
        with self._lock:
            return self._current().attached_definitions()

    def _close(self) -> None:
        proof, scratch = self._proof, self._scratch
        self._proof, self._scratch, self._digest = None, None, ""
        try:
            if proof is not None:
                proof.__exit__()
        finally:
            if scratch is not None:
                scratch.cleanup()

    def close(self) -> None:
        with self._lock:
            self._close()
