"""The hosted world handle: the shipped world vocabulary, backed by one world's own postgres.

`GeneratedWorld` is a database an agent's own generated handlers reach through `Db`. A hosted
world has no handlers to generate — the tables are whatever the agent's own migrations made, and
what a scenario needs is the same six-verb surface (`state`, `put`, `change`, `drop`, `call`,
`query`) built directly on the store, with nothing to adopt or reimplement per agent. Everything
this module refuses, it refuses before the database sees it, so a scenario's mistake reads as a
message naming what it did wrong rather than a `KeyError` or a driver traceback three layers down.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .errors import (
    WorldQueryRejected,
    WorldReadOnly,
    WorldReservedName,
    WorldStateTooLarge,
    WorldUnavailable,
    WorldUsageError,
)
from .runtime import Call
from .stores.postgres import PostgresStore

# The harness's own isolation canary. It exists to prove worlds are really separate from each
# other, never to hold scenario data, so scenario code is never allowed to see or touch it.
CONFORMANCE_TABLE = "_alk_conformance"

# Measured once, at baseline freeze, by the provisioner — never recomputed here. A table over
# this stays over it for the whole run; nothing a scenario does can move which tables raise.
STATE_ROW_CAP = 5000

# The only statement shapes `query()` accepts. Anything else is refused before it reaches the
# database's own read-only transaction, so a statement that was never going to be allowed fails
# on a message naming why rather than a lock error from three layers down.
_READ_KEYWORDS = {"select", "with", "values"}

# The read-only view's fallback answer is restricted to exactly this vocabulary, so a genuinely
# unknown attribute — a capability probe, a dunder, a typo — still reads as a plain
# `AttributeError` instead of masquerading as a write refusal.
_WRITE_VERBS = frozenset({"put", "change", "drop", "call"})


class HostedWorld:
    """The `World` surface, backed by this scenario's own logical postgres database.

    One handle per scenario, over one short-lived autocommit connection per operation — nothing
    held open, which is what lets `reset` drop the database out from under a discarded world.
    `world_index` and `rng` are plain data: the index is for diagnostics only, and the generator
    is the only sanctioned source of randomness scenario code may use.
    """

    def __init__(
        self,
        store: PostgresStore,
        world_index: int,
        rng: random.Random,
        baseline_row_counts: Mapping[str, int],
    ) -> None:
        """Wrap `store` as the `World` surface for one scenario.

        `baseline_row_counts` is keyed by the bare `pg_tables.tablename` value — no schema
        prefix, no quoting — for every table `public` held when the baseline was frozen. A
        visible table missing from this map has no measured cap to enforce, so the coverage
        check below fails construction outright rather than waiting for a scenario's first
        access to discover a provisioning gap through its own retry.
        """
        self._store = store
        self.world_index = world_index
        self.rng = rng
        # Row counts as they stood when the baseline was frozen, keyed by table. Never
        # re-measured: a live count would make the cap depend on what a scenario already wrote,
        # and the whole point is that it is decided before any scenario runs.
        self._baseline_row_counts = dict(baseline_row_counts)
        self._require_baseline_coverage(self._visible_tables())

    # -- reading ------------------------------------------------------------------------------

    def state(self, table: str | None = None) -> dict[str, list[dict[str, Any]]]:
        """A snapshot of the public schema, or of one table in it.

        Bare `state()` leaves an over-cap table out of the snapshot rather than raising through
        it — one seeded audit table must not make the primary read verb inert for the whole
        run. A table nothing measured at baseline freeze gets the same treatment: the only way
        one exists is a table the agent under test created since construction, and nothing it
        does during a call may decide whether bare `state()` raises. Naming either kind of table
        explicitly (`state("big_table")`) still raises: the exclusion is a property of the
        snapshot, not a way to read the table around its own cap. If every visible table is
        over-cap or unmeasured, the exclusion would leave the snapshot `{}` — the one thing
        state() must never return, since an empty snapshot reads as an observation and makes a
        negative check pass on a world nobody actually looked at — so that case raises too,
        naming the tables it would have excluded. The exclusion itself happens at the read: the
        store is asked for only the included tables, not for every table with the excluded ones
        thrown away afterward, so one huge seeded table can no longer make every bare `state()`
        pay to materialise and discard rows nobody asked to see.
        """
        self._reject_reserved(table)
        if table is not None:
            visible = self._visible_tables()
            self._require_nonempty_schema(visible)
            if table not in visible:
                raise WorldUsageError(
                    f"{table!r} is not a table in this world; it holds {sorted(visible)}."
                )
            count = self._row_count(table)
            if count > STATE_ROW_CAP:
                raise WorldStateTooLarge(
                    f"{table!r} held {count} rows when the baseline "
                    f"was frozen, over the {STATE_ROW_CAP}-row cap; state() will not read it "
                    "back."
                )
            return {table: self._store.table(table)}

        names = self._visible_tables()
        self._require_nonempty_schema(names)
        # A table nothing measured at freeze is treated the same as an over-cap one here, not
        # routed through `_row_count`'s typed refusal — that refusal is for a scenario naming a
        # table explicitly; the bare snapshot must not go unavailable over a table the agent
        # itself created since construction (the only actor besides the provisioner that can).
        included = [
            name
            for name in names
            if name in self._baseline_row_counts
            and self._baseline_row_counts[name] <= STATE_ROW_CAP
        ]
        if not included:
            raise WorldStateTooLarge(
                f"every table this world holds — {sorted(names)} — is over the "
                f"{STATE_ROW_CAP}-row cap or was never measured at baseline freeze; state() "
                "will not return {} in their place."
            )
        # One connection, asked for only the included tables — a bare state() used to open a
        # fresh connection per table (and a second one just to look up its primary key) to read
        # every table including the over-cap ones, then throw the over-cap rows away here.
        return self._store.state(only=included)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """The read escape hatch: one statement, on a transaction that cannot write.

        The database's own read-only transaction is the actual guard; this token check only
        makes the common mistake — a stray write, a second statement — fail with a reason
        attached instead of a lock error from underneath.
        """
        _reject_unless_read(sql)
        return self._store.query(sql, tuple(params))

    # -- writing ------------------------------------------------------------------------------

    def put(self, collection: str, record: Mapping[str, Any], *, key: str = "") -> dict[str, Any]:
        """Insert one record; return exactly what the table stored, generated key included.

        `key` exists only to keep this signature a superset of `GeneratedWorld.put`; a hosted
        table already knows its own key — the column its own migrations gave it. Scenario
        authoring has emitted both ``key=<primary-key column>`` and
        ``key=<primary-key value>`` for table-backed worlds, while ``GeneratedWorld`` harmlessly
        ignores either hint. Accept both redundant spellings only when the table has one primary
        key and the record contains it; the value spelling must exactly equal the record's key.
        Continue rejecting arbitrary values and non-primary columns so a mapping-style setup
        cannot silently acquire different semantics after moving to hosted Postgres.
        """
        self._reject_reserved(collection)
        if collection not in self._visible_tables():
            raise WorldUsageError(
                f"{collection!r} is not a table in this world; hosted worlds cannot invent "
                "one, so put() only reaches what the agent's own migrations made."
            )
        if key:
            primary_key = self._primary_key_order(collection)
            primary_key_column = primary_key[0] if len(primary_key) == 1 else ""
            names_primary_key = key == primary_key_column and key in record
            matches_primary_key_value = (
                bool(primary_key_column)
                and primary_key_column in record
                and key == str(record[primary_key_column])
            )
            if not (names_primary_key or matches_primary_key_value):
                raise WorldUsageError(
                    "a hosted table's key is the table's own; key= is accepted only when it "
                    "names the table's single-column primary key or exactly matches that key's "
                    "value in the record."
                )
        return self._store.add(collection, dict(record))

    def change(
        self, collection: str, key: str, changes: Mapping[str, Any], *, by: str = ""
    ) -> int:
        """Update matching records; return how many changed."""
        self._reject_reserved(collection)
        if collection not in self._visible_tables():
            raise WorldUsageError(
                f"{collection!r} is not a table in this world; hosted worlds cannot invent "
                "one, so change() only reaches what the agent's own migrations made."
            )
        by = by or self._resolve_by(collection)
        if by not in self._table_columns(collection):
            raise WorldUsageError(
                f"change({collection!r}, {key!r}, ...) was given by={by!r}, which is not a "
                f"column of {collection!r}."
            )
        return self._store.amend(collection, key, dict(changes), by=by)

    def drop(self, collection: str, key: str = "", *, by: str = "") -> int:
        """Delete matching records, or every record when `key` is empty; return the count."""
        self._reject_reserved(collection)
        if collection not in self._visible_tables():
            raise WorldUsageError(
                f"{collection!r} is not a table in this world; hosted worlds cannot invent "
                "one, so drop() only reaches what the agent's own migrations made."
            )
        if key:
            by = by or self._resolve_by(collection)
            if by not in self._table_columns(collection):
                raise WorldUsageError(
                    f"drop({collection!r}, {key!r}) was given by={by!r}, which is not a "
                    f"column of {collection!r}."
                )
        return self._store.remove(collection, key, by=by)

    def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> Call:
        """Play one of the agent's own tools against this world.

        Not implemented: the `http_tool` evidence seam's wire format is not pinned anywhere in
        the contracts yet, and guessing at a shape here would ship one nobody agreed to and
        scenario code would end up depending on. Raising unconditionally is the honest answer
        until the evidence layer pins it — run-time scenario code should not need this anyway,
        since `setup` already runs at proof time and the data verbs cover the rest.
        """
        raise WorldUnavailable(
            "the http_tool shim wire format is not yet pinned by the contracts — report, "
            "don't guess."
        )

    def read_only(self) -> "ReadOnlyWorld":
        """The view `ready` and `check` run against: every write verb refuses outright.

        A check able to write could not be told apart from one that quietly repaired what it was
        supposed to be grading, so this is what stands between those two functions and the real
        handle.
        """
        return ReadOnlyWorld(self)

    # -- internal -----------------------------------------------------------------------------

    def _reject_reserved(self, collection: str | None) -> None:
        if collection == CONFORMANCE_TABLE:
            raise WorldReservedName(
                f"{collection!r} is the harness's own conformance canary, not scenario data; "
                "it never appears to scenario code."
            )

    def _require_nonempty_schema(self, visible: list[str]) -> None:
        if not visible:
            raise WorldUnavailable(
                "this world's public schema holds no tables to observe; a postgres world with "
                "nothing in it is not a world state() can honestly report on."
            )

    def _require_baseline_coverage(self, names: list[str]) -> None:
        """Refuse rather than assume a table the baseline never measured is under the cap.

        A missing entry used to default to a row count of 0, which would let a table nobody
        measured at freeze read back as though it were known to be small; failing loud here is
        the whole point of deciding the cap before any scenario runs instead of guessing at it.
        Called once, from `__init__`: the table set and the baseline are both fixed for the
        life of the handle, so checking again on every later access would only repeat a answer
        construction already gave.
        """
        missing = [name for name in names if name not in self._baseline_row_counts]
        if missing:
            raise WorldUnavailable(
                f"the baseline row counts this world was built with never measured "
                f"{sorted(missing)}; state()'s cap cannot be decided for a table nobody "
                "measured at freeze."
            )

    def _row_count(self, table: str) -> int:
        """This table's baseline row count, or the same typed refusal `_require_baseline_coverage`
        gives a construction-time gap — for a table that only shows up afterward, so it never
        reaches the caller as a bare `KeyError`.
        """
        try:
            return self._baseline_row_counts[table]
        except KeyError:
            raise WorldUnavailable(
                f"the baseline row counts this world was built with never measured "
                f"{table!r}; state()'s cap cannot be decided for a table nobody measured at "
                "freeze."
            ) from None

    def _visible_tables(self) -> list[str]:
        rows = self._store.query(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
        )
        return [row["tablename"] for row in rows if row["tablename"] != CONFORMANCE_TABLE]

    def _table_columns(self, table: str) -> set[str]:
        rows = self._store.query(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s",
            (table,),
        )
        return {row["column_name"] for row in rows}

    def _primary_key_order(self, table: str) -> list[str]:
        """This table's primary key columns, in index order.

        Joined through `pg_class`/`pg_namespace` rather than a `%s::regclass` cast over an
        f-string, so a table name is only ever a bound value — an embedded `"` (a table created
        as `CREATE TABLE "we""ird" (...)`) is just a character in that value instead of
        something a regclass cast has to parse.
        """
        rows = self._store.query(
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
        )
        return [row["attname"] for row in rows]

    def _resolve_by(self, table: str) -> str:
        """The column to key `change`/`drop` on when scenario code did not name one.

        A single-column primary key is the only case unambiguous enough to guess; no primary
        key at all, or a composite one, means the store genuinely cannot tell which column
        `key` names, so the scenario has to say.
        """
        columns = self._primary_key_order(table)
        if len(columns) == 1:
            return columns[0]
        raise WorldUsageError(
            f"change/drop on {table!r} needs by=<column>; it has no single-column primary "
            "key to default to."
        )


class ReadOnlyWorld:
    """A `World` whose write verbs are refused before they reach `HostedWorld` at all."""

    def __init__(self, world: HostedWorld) -> None:
        # Name-mangled so code outside this class reaching for `._world` cannot casually
        # recover the writable handle a "read-only" view exists to stand in front of.
        self.__world = world
        self.world_index = world.world_index
        self.rng = world.rng

    def state(self, table: str | None = None) -> dict[str, list[dict[str, Any]]]:
        return self.__world.state(table)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        return self.__world.query(sql, params)

    def put(self, collection: str, record: Mapping[str, Any], *, key: str = "") -> dict[str, Any]:
        raise WorldReadOnly(
            f"put({collection!r}, ...) reached a read-only handle; ready() and check() only "
            "ever observe a run."
        )

    def change(
        self, collection: str, key: str, changes: Mapping[str, Any], *, by: str = ""
    ) -> int:
        raise WorldReadOnly(
            f"change({collection!r}, {key!r}, ...) reached a read-only handle; ready() and "
            "check() only ever observe a run."
        )

    def drop(self, collection: str, key: str = "", *, by: str = "") -> int:
        raise WorldReadOnly(
            f"drop({collection!r}, {key!r}) reached a read-only handle; ready() and check() "
            "only ever observe a run."
        )

    def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> Call:
        raise WorldReadOnly(
            f"call({name!r}, ...) reached a read-only handle; ready() and check() only ever "
            "observe a run."
        )

    def __getattr__(self, name: str) -> Any:
        """A write verb `HostedWorld` grows later, with no override here yet, still reads as
        `WorldReadOnly` rather than a plain `AttributeError` indistinguishable from a typo.

        Restricted to that known vocabulary and never to dunders: this repo's own runner code
        reaches for `hasattr(world, "forward")` and `getattr(world, "runtime_tools", set())` on
        world objects, and both only work through the ordinary `AttributeError` those tools
        expect from a name that is simply not there, not from a refusal that happens to look
        like one.
        """
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name in _WRITE_VERBS:
            raise WorldReadOnly(
                f"{name!r} reached a read-only handle; ready() and check() only ever observe "
                "a run."
            )
        raise AttributeError(name)


def _is_word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


_DOLLAR_TAG = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)?\$")


def _dollar_quote_end(sql: str, start: int) -> int | None:
    """The index just past the matching closing `$tag$`, if `sql[start:]` opens one.

    `None` if `start` is not a dollar-quote opener at all — a `$1` parameter placeholder, or a
    bare `$`, never matches the tag grammar and is left for the caller to treat as an ordinary
    character. An opener with no matching close consumes to the end of the string, the same
    fate an unterminated `'`/`"` span already gets below.
    """
    opener = _DOLLAR_TAG.match(sql, start)
    if opener is None:
        return None
    delimiter = opener.group(0)
    end = sql.find(delimiter, opener.end())
    return len(sql) if end == -1 else end + len(delimiter)


def _blank(sql: str, quote_chars: str) -> str:
    """`sql` with every comment blanked, plus any quoted span opened by a character in
    `quote_chars`, plus every dollar-quoted `$tag$...$tag$` span.

    Two callers need two different blindnesses: the shape check does not care what a string
    literal's characters are, so it blanks both quote styles; the reserved-name check must not
    let a quoted identifier hide inside a span it no longer looks at, so it blanks only string
    literals and leaves double-quoted identifiers as text. Dollar-quoting is blanked for both
    callers regardless of `quote_chars` — it is never an identifier, only ever a literal, and a
    `'` or `;` sitting inside one is exactly what both callers must not see as SQL.

    A `'` immediately after a bare `E`/`e` opens an escape string, where a `\\` escapes whatever
    follows it the same way doubling the quote does. Missing that let a `\\'` inside one close
    the literal early: the real closing quote right after it then read as opening a fresh span,
    and everything up to the next quote — semicolon, second statement and all — vanished into
    it as though it were still part of the string.
    """
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        if sql.startswith("--", i):
            end = sql.find("\n", i)
            i = n if end == -1 else end + 1
            continue
        if sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        char = sql[i]
        if char == "$":
            dollar_end = _dollar_quote_end(sql, i)
            if dollar_end is not None:
                i = dollar_end
                out.append(" ")
                continue
        if char in quote_chars:
            escapes = (
                char == "'"
                and i > 0
                and sql[i - 1] in "Ee"
                and (i == 1 or not _is_word_char(sql[i - 2]))
            )
            i += 1
            while i < n:
                if escapes and sql[i] == "\\":
                    i += 2
                    continue
                if sql[i] == char:
                    if sql[i : i + 2] == char * 2:  # an escaped quote inside the literal
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            out.append(" ")
            continue
        out.append(char)
        i += 1
    return "".join(out)


def _sql_skeleton(sql: str) -> str:
    """`sql` with every comment and string literal blanked out to a single space.

    The token check only needs to know what kind of statement this is and how many of them
    there are; without this a semicolon or the word FOR inside a quoted value would count as
    SQL, and the check would end up rejecting a value instead of a statement.
    """
    return _blank(sql, "'\"")


def _names_the_reserved_table(sql: str) -> bool:
    """Whether `_alk_conformance` appears anywhere as an identifier, quoted or not.

    Blanked for string literals only, deliberately not double-quoted identifiers — an
    identifier is exactly where this name could hide from a check that blanked those away too.
    The boundary either side of the name is `\\w`, not a quote: a `"` sitting right against it
    is exactly the character that must not shield it, or `"_alk_conformance"` would read as
    hidden the same way a comment or a literal already is.
    """
    identifiers = _blank(sql, "'")
    pattern = rf"(?<!\w){re.escape(CONFORMANCE_TABLE)}(?!\w)"
    return re.search(pattern, identifiers, re.IGNORECASE) is not None


def _reject_unless_read(sql: str) -> None:
    body = _sql_skeleton(sql).strip()
    if not body:
        raise WorldQueryRejected("query() was given nothing to run.")
    unterminated = body[:-1].strip() if body.endswith(";") else body
    if ";" in unterminated:
        raise WorldQueryRejected("query() runs one statement; this text holds more than one.")
    leading = re.match(r"[A-Za-z_]+", unterminated)
    word = leading.group(0).lower() if leading else ""
    if word not in _READ_KEYWORDS:
        raise WorldQueryRejected(
            f"query() only reads: it takes SELECT, WITH or VALUES, not {word or sql[:20]!r}."
        )
    if re.search(r"\bfor\s+(update|share)\b", unterminated, re.IGNORECASE):
        raise WorldQueryRejected(
            "query() runs on a read-only transaction; FOR UPDATE/FOR SHARE lock rows for a "
            "write that can never follow."
        )
    if _names_the_reserved_table(sql):
        raise WorldQueryRejected(
            f"query() refuses to name {CONFORMANCE_TABLE!r}; it is the harness's own "
            "conformance canary, not scenario data."
        )
