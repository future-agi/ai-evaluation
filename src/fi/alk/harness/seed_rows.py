"""The rows the agent's own seed files insert, read statically.

Only what a world can be held to without running the agent's database: for each table a seed
fills, the primary-key values it inserts. Cells that are not plain literals are left unread.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from .source_data_invariants import source_files

_INSERT = re.compile(
    r"\bINSERT\s+(?:OR\s+\w+\s+)?INTO\s+(?:\"?\w+\"?\.)?\"?([A-Za-z_]\w*)\"?"
    r"\s*\(([^()]*)\)\s*VALUES\s*",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_INTEGER = re.compile(r"[+-]?\d+")
_CAST = re.compile(r"::\s*\w+(?:\s*\(\s*\d+(?:\s*,\s*\d+)?\s*\))?\s*$")


def seed_files(source: Path) -> list[tuple[str, Path]]:
    """The agent's seed SQL files: named for seeding, outside its tests."""
    return [
        (name, path)
        for name, path in source_files(Path(source)).items()
        if name.endswith(".sql")
        and "seed" in name.lower()
        and not any(part.lower().startswith("test") for part in Path(name).parts)
    ]


def _uncommented(sql: str) -> str:
    out: list[str] = []
    at, quoted = 0, False
    while at < len(sql):
        char = sql[at]
        if quoted:
            out.append(char)
            quoted = char != "'"
        elif char == "'":
            out.append(char)
            quoted = True
        elif sql.startswith("--", at):
            end = sql.find("\n", at)
            at = len(sql) if end < 0 else end
            continue
        elif sql.startswith("/*", at):
            end = sql.find("*/", at + 2)
            at = len(sql) if end < 0 else end + 2
            continue
        else:
            out.append(char)
        at += 1
    return "".join(out)


def _tuples(sql: str, at: int) -> list[list[str]]:
    """The VALUES tuples starting at ``at``, each cell as its raw text."""
    rows: list[list[str]] = []
    while True:
        while at < len(sql) and sql[at].isspace():
            at += 1
        if at >= len(sql) or sql[at] != "(":
            return rows
        cells, cell, depth, quoted = [], [], 1, False
        at += 1
        while at < len(sql) and depth:
            char = sql[at]
            if quoted:
                quoted = char != "'"
            elif char == "'":
                quoted = True
            elif char in "([":
                depth += 1
            elif char in ")]":
                depth -= 1
            if depth == 1 and not quoted and char == ",":
                cells.append("".join(cell))
                cell = []
            elif depth:
                cell.append(char)
            at += 1
        cells.append("".join(cell))
        rows.append(cells)
        while at < len(sql) and sql[at].isspace():
            at += 1
        if at >= len(sql) or sql[at] != ",":
            return rows
        at += 1


def _literal(cell: str) -> str | None:
    """A plain literal's value as text, or None for anything else."""
    text = _CAST.sub("", cell.strip()).strip()
    if len(text) >= 2 and text[0] == text[-1] == "'":
        inner = text[1:-1]
        return None if "'" in inner.replace("''", "") else inner.replace("''", "'")
    if _INTEGER.fullmatch(text) and len(text.lstrip("+-0")) <= 19:
        value = int(text)
        # Exact where SQLite keeps an integer: a float would round keys above 2**53.
        if -(2**63) <= value < 2**63:
            return normal(value)
    if _NUMBER.fullmatch(text):
        return normal(float(text))
    return {"TRUE": "1", "FALSE": "0"}.get(text.upper())


def normal(value: object) -> str:
    """One text form for a key value, so 38, 38.0 and '38' compare equal."""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(value)
    return str(value)


def seed_rows(source: Path) -> dict[str, list[tuple[str, dict[str, str | None]]]]:
    """Per table (lower-cased), the rows the seed inserts: (file, {column: literal or None})."""
    found: dict[str, list[tuple[str, dict[str, str | None]]]] = {}
    for name, path in seed_files(source):
        try:
            sql = _uncommented(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
        for insert in _INSERT.finditer(sql):
            columns = [
                one.strip().strip('"').lower() for one in insert.group(2).split(",")
            ]
            for cells in _tuples(sql, insert.end()):
                if len(cells) == len(columns):
                    row = {col: _literal(cell) for col, cell in zip(columns, cells)}
                    found.setdefault(insert.group(1).lower(), []).append((name, row))
    return found


def primary_key(connection: sqlite3.Connection, table: str) -> str | None:
    """The table's single-column primary key, or None."""
    keys = [
        row[1]
        for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
        if row[5]
    ]
    return str(keys[0]) if len(keys) == 1 else None


def missing_seed_rows(
    connection: sqlite3.Connection, source: Path
) -> dict[str, tuple[str, list[str], list[str]]]:
    """Seed rows a SQLite world lacks: {table: (key column, missing keys, seed files)}.

    Only tables the world has. By primary key where the world has one and the seed lists it;
    a table with no usable key is only held to having rows when the seed puts rows in it.
    """
    seeded = seed_rows(source)
    tables = {
        str(row[0]).lower(): str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    missing: dict[str, tuple[str, list[str], list[str]]] = {}
    for lowered, rows in seeded.items():
        table = tables.get(lowered)
        if table is None:
            continue
        files = sorted({name for name, _row in rows})
        column = primary_key(connection, table)
        if column is None or any(column.lower() not in row for _name, row in rows):
            if not connection.execute(f'SELECT 1 FROM "{table}" LIMIT 1').fetchone():
                missing[table] = ("", [f"{len(rows)} row(s)"], files)
            continue
        held = {
            normal(value)
            for (value,) in connection.execute(f'SELECT "{column}" FROM "{table}"')
        }
        wanted = [row[column.lower()] for _name, row in rows]
        absent = sorted({key for key in wanted if key is not None} - held)
        if absent:
            missing[table] = (column, absent, files)
    return missing


def only_seed_rows(connection: sqlite3.Connection, source: Path, table: str) -> bool:
    """Whether every row in the table is one the agent's seed inserts, by primary key."""
    column = primary_key(connection, table)
    if column is None:
        return False
    keys = {
        row.get(column.lower())
        for _name, row in seed_rows(source).get(table.lower(), [])
    }
    held = [
        normal(value)
        for (value,) in connection.execute(f'SELECT "{column}" FROM "{table}"')
    ]
    return bool(held) and all(key in keys for key in held)
