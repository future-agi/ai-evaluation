"""What the contract has to carry before an environment can be stood up.

The build stage cannot provision from prose. It needs to know which engine to run and how to
be the thing the agent already connects to -- and it has to get both without anyone editing
the agent, which is why what is recorded is the agent's existing expectations rather than a
change to be made to it.
"""

from __future__ import annotations

from fi.alk.harness.contract import AgentContract, Dependency, Reached


def contract(**kwargs) -> AgentContract:
    return AgentContract(agent="orders", **kwargs)


# --- is there enough here to stand something up? ----------------------------------------


def test_an_engine_with_a_dsn_variable_is_provisionable() -> None:
    one = Dependency(
        name="orders db",
        kind="datastore",
        engine="postgres",
        version="16",
        reached=Reached(dsn_env="DATABASE_URL"),
    )
    assert one.provisionable()


def test_an_engine_reached_only_by_a_config_key_is_provisionable() -> None:
    one = Dependency(
        name="orders db",
        engine="clickhouse",
        reached=Reached(config_key="database.url"),
    )
    assert one.provisionable()


def test_hardcoded_values_are_a_seam_rather_than_a_dead_end() -> None:
    """A hardcoded host is redirected by a network alias, not by editing the agent."""
    one = Dependency(
        name="orders db",
        engine="postgres",
        reached=Reached(host="db.internal", port=5432, database="app"),
    )
    assert one.provisionable()


def test_an_engine_with_no_seam_at_all_is_not_provisionable() -> None:
    one = Dependency(name="orders db", engine="postgres")
    assert not one.provisionable()
    assert not one.reached.has_seam()


def test_a_seam_with_no_engine_is_not_provisionable() -> None:
    """Knowing where to point it is useless without knowing what to run."""
    one = Dependency(name="orders db", reached=Reached(dsn_env="DATABASE_URL"))
    assert not one.provisionable()


def test_a_dependency_that_says_nothing_still_loads() -> None:
    """Every existing contract predates these fields and must keep working."""
    one = Dependency(name="OpenAI TTS", kind="service", what="speech")
    assert one.engine == ""
    assert not one.provisionable()


# --- and does the build stage get told? --------------------------------------------------


def rendered(one: Dependency) -> str:
    return contract(dependencies=[one]).brief()


def test_the_engine_and_its_seam_are_said_together() -> None:
    text = rendered(
        Dependency(
            name="orders db",
            engine="postgres",
            version="16",
            reached=Reached(dsn_env="DATABASE_URL"),
        )
    )
    assert "stand up postgres 16" in text
    assert "$DATABASE_URL" in text


def test_what_the_agent_expects_is_rendered_as_something_to_match() -> None:
    text = rendered(
        Dependency(
            name="orders db",
            engine="postgres",
            reached=Reached(host="db.internal", port=5432, database="app", user="svc"),
        )
    )
    assert "build it to match" in text
    assert "host db.internal" in text
    assert "port 5432" in text
    assert "user svc" in text


def test_an_engine_with_no_seam_is_called_out_where_it_will_be_read() -> None:
    """Better said here than discovered later, when the reason is much harder to see."""
    text = rendered(Dependency(name="orders db", engine="postgres"))
    assert "NO CONFIGURATION SEAM" in text


def test_a_plain_dependency_renders_as_it_always_did() -> None:
    text = rendered(Dependency(name="OpenAI TTS", kind="service", what="speech"))
    assert "OpenAI TTS (service): speech" in text
    assert "stand up" not in text


def test_the_contract_says_the_agent_is_never_edited() -> None:
    """The instruction lives where the build stage reads it, not only in a skill file."""
    text = rendered(Dependency(name="orders db", engine="postgres"))
    assert "code is never edited" in text


# --- and the thing we refuse to write down ------------------------------------------------


def test_a_password_has_nowhere_to_go() -> None:
    """A contract is written to disk and read by people; a secret in it outlives the run.

    Only where the value comes from is recorded, never the value.
    """
    assert not hasattr(Reached(), "password")
    assert Reached(password_from="PGPASSWORD").password_from == "PGPASSWORD"


def test_round_trips_through_json() -> None:
    one = Dependency(
        name="orders db",
        engine="postgres",
        version="16",
        reached=Reached(dsn_env="DATABASE_URL", host="db.internal", port=5432),
    )
    again = Dependency.model_validate(one.model_dump())
    assert again.reached.host == "db.internal"
    assert again.reached.port == 5432
    assert again.provisionable()
