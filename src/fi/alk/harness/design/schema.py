"""What a designed world's SQLite schema may hold, given that hosted runs seed it into PostgreSQL.

`world_seed_sql` converts tables, column types, primary keys, NOT NULL, literal defaults and
non-partial unique indexes. It drops or rewrites everything else without a word, so a world that
relies on it behaves one way while it is designed and another in runs. A design may not add any
of it. What a template already had is listed as inherited; its handlers enforce those rules on
both engines.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from ..bundle_author_v2 import _safe_sql_default, _sqlite_type

_TYPES = frozenset({"TEXT", "INTEGER", "REAL", "NUMERIC", "BOOLEAN"})
_CHECK = re.compile(r"\bCHECK\s*\(", re.IGNORECASE)


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _checks(sql: str) -> list[str]:
    """Each `CHECK (...)` expression in a table's definition, whitespace-normalized."""
    found = []
    for match in _CHECK.finditer(sql):
        start, depth = match.end() - 1, 0
        for at in range(start, len(sql)):
            depth += {"(": 1, ")": -1}.get(sql[at], 0)
            if depth == 0:
                found.append(" ".join(sql[start : at + 1].split()))
                break
    return found


def _table(connection: sqlite3.Connection, table: str, sql: str) -> list[str]:
    features = []
    quoted = _quoted(table)
    for _cid, column, declared, _not_null, default, _pk, hidden in connection.execute(
        f"PRAGMA table_xinfo({quoted})"
    ):
        if hidden in (2, 3):
            features.append(f"{table}.{column}: generated column")
        if (declared or "").split("(")[0].strip().upper() not in _TYPES:
            features.append(f"{table}.{column}: type {declared or 'none'}")
        # Dropping DEFAULT NULL changes nothing; a literal survives; anything else does not.
        if (
            default is not None
            and str(default).strip().upper() != "NULL"
            and _safe_sql_default(default, sql_type=_sqlite_type(declared or ""))
            in (None, "now()")
        ):
            features.append(f"{table}.{column}: default {default}")
    for row in connection.execute(f"PRAGMA foreign_key_list({quoted})"):
        parent, column, parent_column = row[2], row[3], row[4]
        features.append(
            f"{table}.{column}: references {parent}({parent_column or 'rowid'})"
        )
    features.extend(f"{table}: check {check}" for check in _checks(sql))
    for _seq, index, unique, origin, partial in connection.execute(
        f"PRAGMA index_list({quoted})"
    ):
        if origin == "pk":
            continue
        columns = [
            row[2] for row in connection.execute(f"PRAGMA index_info({_quoted(index)})")
        ]
        if partial:
            features.append(f"{table}: partial index {index}")
        elif not unique:
            features.append(f"{table}: index {index}")
        elif None in columns:
            features.append(f"{table}: expression index {index}")
    return features


def unseeded_schema(database: Path) -> list[str]:
    """Each schema feature of a world's database that its PostgreSQL seed drops or rewrites."""
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        features = []
        for kind, name, sql in connection.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite\\_%' ESCAPE '\\' ORDER BY type, name"
        ).fetchall():
            if kind in ("view", "trigger"):
                features.append(f"{kind} {name}")
            elif kind == "table":
                features.extend(_table(connection, name, sql or ""))
        return sorted(features)
    finally:
        connection.close()
