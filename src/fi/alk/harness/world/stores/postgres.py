"""Postgres, as the worked example of what an engine has to supply.

This is not "the database the harness supports". It is the reference: when the build stage
finds an agent on ClickHouse or MySQL or DuckDB, what it writes is a class this shape, and
what it has to work out is only what is in this file below ``boot_env`` -- how to reach the
engine, how to read what it holds, and how to put that back. Starting a container, finding a
free port, waiting for the thing to genuinely answer and not leaking it afterwards are all in
``ContainerStore`` and are never rewritten.

Nothing here knows what the agent's tools do. The agent keeps its own client, its own SQL and
its own migrations; the only thing that changed is the host on the far end of its DSN. The
schema is not invented either -- the build stage runs the agent's own migrations through
``apply``, so the tables are the agent's tables, spelled the way the agent spells them. A
schema we wrote ourselves would be a guess, and every check written against it would inherit
the guess.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..errors import WorldQueryRejected
from . import Held, Snapshot, StoreError
from .container import ContainerStore, docker

SCHEMA = "schema.sql"


def _psycopg() -> Any:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise StoreError(
            "psycopg is not installed, so a Postgres store cannot be read. Install it with "
            "`uv sync --extra harness-stores`."
        ) from exc
    return psycopg


class PostgresStore(ContainerStore):
    """A Postgres container the agent under test is pointed at."""

    engine = "postgres"
    image = "postgres:16"
    container_port = 5432
    boot_env = {
        "POSTGRES_USER": "{user}",
        "POSTGRES_PASSWORD": "{password}",
        "POSTGRES_DB": "{database}",
    }

    # -- how to reach it -------------------------------------------------------------

    def dsn(self) -> str:
        external = os.environ.get("ALK_POSTGRES_DSN", "").strip()
        if external:
            return external
        host, port = self.address()
        return f"postgresql://{self.user}:{self.password}@{host}:{port}/{self.database}"

    def probe(self) -> None:
        """Really connect. A running container is not yet a database that listens."""
        with _psycopg().connect(self.dsn(), connect_timeout=3) as connection:
            connection.execute("SELECT 1")

    def _connect(self) -> Any:
        """A short-lived autocommit connection.

        Deliberately not pooled and never held open. An idle transaction of ours would block
        the ``TRUNCATE`` in ``restore``, and a reset that hangs on the harness's own connection
        is a very expensive thing to debug.
        """
        return _psycopg().connect(self.dsn(), autocommit=True)

    # -- how to read what it holds ---------------------------------------------------

    def apply(self, script: str) -> None:
        """Run whatever was handed in: the agent's migrations, or its seed."""
        if not script.strip():
            return
        with self._connect() as connection:
            connection.execute(script)
        # Remembered because the snapshot holds rows, not DDL. A restore into a fresh
        # container finds no tables, and restoring rows into a schema that is not there
        # quietly restores nothing.
        self.applied.append(script)

    def execute(self, statement: str, params: Sequence[Any] = ()) -> int:
        """Run one mutation and report the rows it actually changed."""
        with self._connect() as connection:
            cursor = connection.execute(statement, tuple(params))
            return max(0, int(cursor.rowcount or 0))

    def query(self, statement: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """Run one read statement, on a connection Postgres itself will not let write.

        Autocommit, with the session's own default flipped to read-only rather than one shared
        ``SET TRANSACTION READ ONLY`` transaction: every statement becomes its own implicit
        read-only transaction, so nothing here is ever held open, and ``reset``'s drop of the
        database can always proceed regardless of what a caller just read.

        An empty ``params`` tuple is passed through as ``None`` rather than as itself: psycopg
        scans for placeholders whenever it is handed anything other than ``None``, and an
        ordinary ``LIKE '%turkey%'`` with nothing to bind then reads its own ``%t`` as an
        unmatched one and raises before the statement ever reaches Postgres.
        """
        with _psycopg().connect(
            self.dsn(), autocommit=True, options="-c default_transaction_read_only=on"
        ) as connection:
            cursor = connection.execute(statement, tuple(params) if params else None)
            columns = [description[0] for description in cursor.description or []]
            seen: set[str] = set()
            for column in columns:
                if column in seen:
                    raise WorldQueryRejected(
                        f"query() returned more than one column named {column!r}; alias one "
                        "of them so a row does not silently lose one under the other."
                    )
                seen.add(column)
            return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    def _tables(self, connection: Any) -> list[str]:
        rows = connection.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
        ).fetchall()
        return [row[0] for row in rows]

    def _primary_key(self, connection: Any, table: str) -> list[str]:
        """The primary key columns, used only to read rows back in a stable order.

        Joined through ``pg_class``/``pg_namespace`` rather than a ``%s::regclass`` cast over an
        f-string, so a table name is only ever a bound value — an embedded ``"`` (a table
        created as ``CREATE TABLE "we""ird" (...)``) is just a character in that value instead
        of something a regclass cast has to parse.
        """
        rows = connection.execute(
            """
            SELECT a.attname
              FROM pg_index i
              JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
              JOIN pg_class c ON c.oid = i.indrelid
              JOIN pg_namespace n ON n.oid = c.relnamespace
             WHERE n.nspname = 'public' AND c.relname = %s AND i.indisprimary
             ORDER BY array_position(i.indkey, a.attnum)
            """,
            (table,),
        ).fetchall()
        return [row[0] for row in rows]

    def _column_types(self, connection: Any, table: str) -> dict[str, str]:
        """Return declared column types so writes use the source schema's representation.

        New managed clusters are always UTF-8, but an externally attached or older SQL_ASCII
        database can return even information-schema text as ``bytes``. Decode the catalogue
        defensively: silently missing every lookup would disable all type adaptation and turn one
        environment detail into a sequence of misleading per-column database failures.
        """
        rows = connection.execute(
            """
            SELECT column_name, data_type
              FROM information_schema.columns
             WHERE table_schema = 'public' AND table_name = %s
            """,
            (table,),
        ).fetchall()

        def text(value: Any) -> str:
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)

        return {text(row[0]): text(row[1]) for row in rows}

    def _select_ordered(self, connection: Any, table: str) -> list[dict[str, Any]]:
        """Every row of one table, ordered by its primary key where it has one.

        Without that order the same data comes back in whatever sequence the heap happens to
        hold it, and a check comparing the first row is reading a coin toss rather than the
        agent's behaviour. Built with ``sql.Identifier`` rather than an f-string because
        ``table`` is a name the harness only just read out of the catalogue, not a literal it
        wrote itself.
        """
        sql = _psycopg().sql
        key = self._primary_key(connection, table)
        statement = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table))
        if key:
            statement += sql.SQL(" ORDER BY ") + sql.SQL(", ").join(
                sql.Identifier(column) for column in key
            )
        cursor = connection.execute(statement)
        columns = [description[0] for description in cursor.description or []]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    def state(
        self, only: Sequence[str] | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """Every table and its rows, in the shape the checks already expect.

        ``only`` narrows the read to the named tables, still inside the one connection — a
        caller that already knows it wants a subset (``HostedWorld`` excluding over-cap tables)
        never pays to read and discard rows for the ones it does not. ``None`` reads every table
        in ``public``, exactly as before.
        """
        with self._connect() as connection:
            tables = self._tables(connection) if only is None else list(only)
            return {table: self._select_ordered(connection, table) for table in tables}

    def table(self, name: str) -> list[dict[str, Any]]:
        """One table's rows, ordered by primary key where it has one.

        The read-side counterpart to ``state()``'s per-table loop, for a caller that wants only
        one of them: still a single connection, so reading one table never costs a second round
        trip just to learn how to order it.
        """
        with self._connect() as connection:
            return self._select_ordered(connection, name)

    # -- how to put it back ----------------------------------------------------------

    def freeze(self) -> Snapshot:
        """Rows and sequence counters, which together are the whole mutable state."""
        with self._connect() as connection:
            counters = {
                row[0]: row[1]
                for row in connection.execute(
                    "SELECT sequencename, last_value FROM pg_sequences "
                    "WHERE schemaname = 'public'"
                ).fetchall()
                if row[1] is not None
            }
        return Snapshot(rows=self.state(), counters=counters)

    def restore(self, snapshot: Snapshot) -> None:
        """Put the data back exactly as the snapshot found it.

        Foreign keys are suspended for the duration rather than the rows being sorted into
        dependency order: the snapshot was taken from a consistent database, so what goes back
        is consistent by construction, and ordering it would be solving a problem we do not
        have. Counters are set last, so the next scenario's first insert gets the id the first
        scenario's did.
        """
        with self._connect() as connection:
            tables = self._tables(connection)
            if not tables:
                return
            listed = ", ".join(f'"{table}"' for table in tables)
            # One statement, so Postgres resolves the dependency order between them itself.
            connection.execute(f"TRUNCATE TABLE {listed} RESTART IDENTITY CASCADE")

            connection.execute("SET session_replication_role = replica")
            try:
                for table, rows in snapshot.rows.items():
                    if not rows or table not in tables:
                        continue
                    columns = list(rows[0])
                    types = self._column_types(connection, table)
                    quoted = ", ".join(f'"{column}"' for column in columns)
                    placeholders = ", ".join(["%s"] * len(columns))
                    statement = (
                        f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders})'
                    )
                    with connection.cursor() as cursor:
                        cursor.executemany(
                            statement,
                            [
                                tuple(
                                    _adapt(row.get(column), types.get(column, ""))
                                    for column in columns
                                )
                                for row in rows
                            ],
                        )
            finally:
                connection.execute("SET session_replication_role = DEFAULT")

            for sequence, value in snapshot.counters.items():
                connection.execute(
                    "SELECT setval(%s, %s, true)", (f'public."{sequence}"', value)
                )

    def save_to(self, path: str | Path) -> None:
        """Save both the records and the DDL a fresh Postgres store needs."""
        Held.save_to(self, path)
        root = Path(path)
        schema = docker(
            "exec",
            self.container,
            "pg_dump",
            "--schema-only",
            "--no-owner",
            "--no-privileges",
            "--username",
            self.user,
            "--dbname",
            self.database,
        )
        # pg_dump can emit psql-only safety commands. The snapshot is replayed through psycopg,
        # so keep SQL and discard client meta-commands.
        schema = "\n".join(
            line for line in schema.splitlines() if not line.startswith("\\")
        )
        (root / SCHEMA).write_text(schema, encoding="utf-8")

    def load_from(self, path: str | Path) -> None:
        root = Path(path)
        schema = root / SCHEMA
        if not schema.exists():
            raise StoreError(f"no saved Postgres schema at {schema}")
        self.apply(schema.read_text(encoding="utf-8"))
        Held.load_from(self, root)

    # -- what a scenario changes -----------------------------------------------------

    def add(self, collection: str, record: Any) -> dict[str, Any]:
        """Insert one record and hand back exactly what Postgres stored.

        ``RETURNING *`` rather than a second read: a caller after the row's generated key (an
        identity column, a default, a trigger) would otherwise have to guess which column that
        is, and a table with no natural way to re-select the row it just inserted could not be
        read back at all.
        """
        columns = list(record)
        sql = _psycopg().sql
        statement = sql.SQL("INSERT INTO {} ({}) VALUES ({}) RETURNING *").format(
            sql.Identifier(collection),
            sql.SQL(", ").join(sql.Identifier(column) for column in columns),
            sql.SQL(", ").join(sql.SQL("%s") for _ in columns),
        )
        with self._connect() as connection:
            types = self._column_types(connection, collection)
            cursor = connection.execute(
                statement,
                tuple(
                    _adapt(record[column], types.get(column, "")) for column in columns
                ),
            )
            stored = cursor.fetchone()
            if stored is None:
                raise StoreError(
                    f"INSERT INTO {collection!r} ... RETURNING * came back with no row; a rule "
                    "or a BEFORE INSERT trigger the agent's own migrations declared can turn an "
                    "insert into a no-op, and put() cannot report a record that was never "
                    "written."
                )
            out = [description[0] for description in cursor.description or []]
            return dict(zip(out, stored, strict=True))

    def amend(self, collection: str, key: str, changes: Any, *, by: str = "") -> int:
        if not by:
            raise StoreError(
                f"{collection} is a table, so changing a record needs the column it is keyed on"
            )
        sql = _psycopg().sql
        statement = sql.SQL("UPDATE {} SET {} WHERE {} = %s").format(
            sql.Identifier(collection),
            sql.SQL(", ").join(
                sql.SQL("{} = %s").format(sql.Identifier(column)) for column in changes
            ),
            sql.Identifier(by),
        )
        with self._connect() as connection:
            types = self._column_types(connection, collection)
            cursor = connection.execute(
                statement,
                (
                    *(
                        _adapt(value, types.get(column, ""))
                        for column, value in changes.items()
                    ),
                    key,
                ),
            )
            return cursor.rowcount

    def remove(self, collection: str, key: str = "", *, by: str = "") -> int:
        if key and not by:
            raise StoreError(
                f"{collection} is a table, so removing one record needs the column it is keyed on"
            )
        sql = _psycopg().sql
        statement = sql.SQL("DELETE FROM {}").format(sql.Identifier(collection))
        if key:
            statement += sql.SQL(" WHERE {} = %s").format(sql.Identifier(by))
        with self._connect() as connection:
            cursor = connection.execute(statement, (key,) if key else ())
            return cursor.rowcount


class AttachedPostgresStore(PostgresStore):
    """A Postgres store already started by the submitted repository's Compose project.

    The record and snapshot operations are exactly the same as ``PostgresStore``. Only ownership
    differs: the harness may inspect and reset this database, but the Compose provisioner owns
    its process and lifecycle, so closing one scenario must never remove the shared container.
    """

    def __init__(self, dsn: str) -> None:
        self._external_dsn = dsn
        self.applied: list[str] = []
        self._started = True

    def dsn(self) -> str:
        return self._external_dsn

    def start(self) -> None:
        self.probe()

    def stop(self) -> None:
        return

    def save_to(self, path: str | Path) -> None:
        # Compose owns the schema and reruns the repository's migrations/initialisers whenever
        # the project is recreated. The harness snapshot therefore owns only mutable rows and
        # counters, avoiding a second generated schema that can drift from the submitted code.
        from . import Held

        Held.save_to(self, path)

    def load_from(self, path: str | Path) -> None:
        from . import Held

        Held.load_from(self, path)


def _adapt(value: Any, data_type: str = "") -> Any:
    """Hand back a value in the form psycopg will write.

    A list in a JSON column must be wrapped, while a list in an ARRAY column must remain a list
    so psycopg emits a native Postgres array. The authored setup crosses JSON boundaries before
    it reaches a store, and some producers consequently leave a structured array as the JSON
    string ``"[]"``. PostgreSQL interprets a bound string as its own array-literal syntax, where
    square brackets mean dimensions rather than values, and rejects it. Decode only an actual
    JSON list for an ARRAY column; arbitrary strings remain arbitrary strings and PostgreSQL can
    enforce the declared element type.

    Scenario setup may likewise represent a boolean as ``0``/``1`` even though PostgreSQL
    deliberately does not implicitly cast a bound smallint to boolean. Normalize only the
    finite, unambiguous boolean vocabulary; using ``bool(value)`` here would silently turn values
    such as ``2`` or ``"disabled"`` into true.
    """
    normalized_type = data_type.strip().lower()
    if value is not None and normalized_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            normalized_value = value.strip().lower()
            if normalized_value in {"true", "t", "1"}:
                return True
            if normalized_value in {"false", "f", "0"}:
                return False
        raise StoreError(
            "a PostgreSQL boolean column received "
            f"{value!r}; expected true/false or the equivalent 1/0"
        )
    if value is not None and normalized_type == "array":
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, str) and value.strip().startswith("["):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as exc:
                raise StoreError(
                    "a PostgreSQL array column received malformed JSON array text "
                    f"{value!r}"
                ) from exc
            if not isinstance(decoded, list):  # pragma: no cover - guarded by '['
                raise StoreError(
                    f"a PostgreSQL array column received {value!r}; expected a JSON list"
                )
            return decoded
    if value is not None and normalized_type in ("json", "jsonb"):
        from psycopg.types.json import Jsonb

        return Jsonb(value)
    return value
