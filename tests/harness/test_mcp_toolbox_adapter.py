import json
import os
from contextlib import contextmanager

import pytest

from fi.alk.harness.mcp_toolbox_adapter import _execute, _signature


@pytest.mark.parametrize(
    "names",
    [
        ["class"],
        ["value", "value"],
        ["_execute"],
        ["_spec"],
        ["_name"],
        ["_database_url"],
        ["_trace_path"],
        ["locals"],
        [" padded "],
        ["ｌｏｃａｌｓ"],
    ],
)
def test_signature_rejects_invalid_or_shadowing_parameters(names):
    with pytest.raises(RuntimeError, match="unsupported Toolbox parameter"):
        _signature([{"name": name, "type": "string"} for name in names])


def test_signature_accepts_source_parameter_names():
    assert (
        _signature(
            [
                {"name": "assignee", "type": "string"},
                {"name": "limit", "type": "integer"},
            ]
        )
        == "assignee: str, limit: int"
    )


def test_execute_preserves_native_parameters_without_database(monkeypatch, tmp_path):
    import psycopg
    from psycopg.adapt import Transformer

    captured = []

    @contextmanager
    def connect(_dsn, *, autocommit, cursor_factory=psycopg.Cursor):
        class Connection:
            description = None
            rowcount = 0

            @contextmanager
            def cursor(self):
                yield self

            def execute(self, statement, parameters):
                query = cursor_factory._query_cls(Transformer())
                query.convert(statement, parameters)
                captured.append((query.query, list(map(bytes, query.params))))

        yield Connection()

    monkeypatch.setattr(psycopg, "connect", connect)
    statement = (
        "SELECT '$1', $$ $2 $$, $tag$ $1 $tag$, '100%', $2, $1, $1 -- $99\n/* $98 */"
    )
    _execute(
        "example",
        {"statement": statement, "parameters": [{"name": "first"}, {"name": "second"}]},
        {"first": "one", "second": "two"},
        database_url="unused",
        trace_path=tmp_path / "trace.jsonl",
    )
    assert captured == [(statement.encode(), [b"one", b"two"])]


def test_source_sql_executes_without_rewriting_literals_or_comments(tmp_path):
    dsn = os.environ.get("ALK_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set ALK_TEST_POSTGRES_DSN for native SQL parameter verification")
    statement = """SELECT '$1' AS literal, $$dollar $2$$ AS dollar,
        $tag$tagged $1$tag$ AS tagged, '100%' AS percent,
        $2::text AS second, $1::text AS first, $1::text AS repeated
        -- ignored $99
        /* ignored $98 /* nested $97 */ */"""
    trace = tmp_path / "trace.jsonl"
    result = _execute(
        "example",
        {"statement": statement, "parameters": [{"name": "first"}, {"name": "second"}]},
        {"first": "one", "second": "two"},
        database_url=dsn,
        trace_path=trace,
    )
    assert json.loads(result) == [
        {
            "literal": "$1",
            "dollar": "dollar $2",
            "tagged": "tagged $1",
            "percent": "100%",
            "second": "two",
            "first": "one",
            "repeated": "one",
        }
    ]
    assert json.loads(trace.read_text())["is_error"] is False
