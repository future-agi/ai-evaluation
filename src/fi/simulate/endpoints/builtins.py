"""Back-compat shim. Builtin endpoint registrations moved to
``fi.simulate.endpoints.profiles`` (slice 3) — importing this module still
triggers registration via that module's import side effect.
"""

from __future__ import annotations

from fi.simulate.endpoints import profiles as _profiles  # noqa: F401

__all__: list[str] = []
