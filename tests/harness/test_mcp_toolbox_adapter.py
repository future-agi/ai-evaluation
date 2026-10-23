from fi.alk.harness.mcp_toolbox_adapter import _bind_sql


def test_bind_sql_preserves_literal_percent_wildcards() -> None:
    statement, values = _bind_sql(
        {
            "parameters": [{"name": "assignee"}],
            "statement": "SELECT * FROM tickets WHERE assignee ILIKE '%' || $1 || '%'",
        },
        {"assignee": "maria@example.com"},
        tool_name="get-tickets-by-assignee",
    )

    assert statement == "SELECT * FROM tickets WHERE assignee ILIKE '%%' || %s || '%%'"
    assert values == ["maria@example.com"]
