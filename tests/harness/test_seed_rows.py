import asyncio
import sqlite3
from pathlib import Path

import pytest

from fi.alk.harness.seed_rows import (
    missing_seed_rows,
    only_seed_rows,
    seed_files,
    seed_rows,
)

_SEED = """\
-- Seed data; a comment's INSERT INTO users (rider_id) VALUES ('rdr_ghost'); is not a row.
/* INSERT INTO users (rider_id) VALUES ('rdr_block'); */
INSERT INTO users (rider_id, phone, rating, active) VALUES
  ('rdr_dana',   '+14155550101', 4.9, TRUE),
  ('rdr_o''neil', '+14155550102', 4.6, FALSE);

INSERT INTO public."places" (place_id, formatted_address, aliases) VALUES
  ('plc_hilton', '333 O''Farrell Street, San Francisco, CA',
     ARRAY['hilton', 'union square, hilton']);

INSERT INTO trips (trip_id, taken_at, fare_total) VALUES
  ('trp_1', now() - interval '3 days', 48.20),
  (nextval('trip_ids'), now(), 1);

INSERT INTO bookings (booking_ref, status, fee) VALUES
  ('UB7F3A2C91'::text, 'matched', 5.00)
  ON CONFLICT (booking_ref) DO NOTHING;
INSERT INTO otp_codes VALUES ('+14155550101', '638204');
"""


def _source(root: Path, seed: str = _SEED) -> Path:
    (root / "db").mkdir(parents=True)
    (root / "db" / "seed.sql").write_text(seed, encoding="utf-8")
    (root / "db" / "schema.sql").write_text(
        "CREATE TABLE users (rider_id TEXT PRIMARY KEY);\n"
        "INSERT INTO users (rider_id) VALUES ('rdr_schema');\n",
        encoding="utf-8",
    )
    (root / "tests").mkdir()
    (root / "tests" / "seed.sql").write_text(
        "INSERT INTO users (rider_id) VALUES ('rdr_test');\n", encoding="utf-8"
    )
    return root


def test_the_seed_files_are_the_seed_named_ones_outside_tests(tmp_path):
    _source(tmp_path)
    (tmp_path / "db" / "seeds").mkdir()
    (tmp_path / "db" / "seeds" / "02_more.sql").write_text("", encoding="utf-8")

    assert [name for name, _path in seed_files(tmp_path)] == [
        "db/seed.sql",
        "db/seeds/02_more.sql",
    ]


def test_seed_inserts_are_read_from_the_shapes_seeds_use(tmp_path):
    rows = seed_rows(_source(tmp_path))

    assert [row for _file, row in rows["users"]] == [
        {
            "rider_id": "rdr_dana",
            "phone": "+14155550101",
            "rating": "4.9",
            "active": "1",
        },
        {
            "rider_id": "rdr_o'neil",
            "phone": "+14155550102",
            "rating": "4.6",
            "active": "0",
        },
    ]
    [(_file, place)] = rows["places"]
    assert place["place_id"] == "plc_hilton" and place["aliases"] is None
    assert place["formatted_address"] == "333 O'Farrell Street, San Francisco, CA"
    assert [row["trip_id"] for _file, row in rows["trips"]] == ["trp_1", None]
    assert rows["trips"][0][1]["fare_total"] == "48.2"
    assert rows["bookings"] == [
        ("db/seed.sql", {"booking_ref": "UB7F3A2C91", "status": "matched", "fee": "5"})
    ]
    assert "otp_codes" not in rows


def _world(*statements: str) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        "CREATE TABLE users (rider_id TEXT PRIMARY KEY, phone TEXT);"
        "CREATE TABLE bookings (booking_ref TEXT PRIMARY KEY, status TEXT);"
        "CREATE TABLE trips (trip_id TEXT, fare_total REAL);" + "".join(statements)
    )
    return connection


def test_missing_rows_are_found_by_key_in_tables_the_world_has(tmp_path):
    source = _source(tmp_path)
    world = _world(
        "INSERT INTO users VALUES ('rdr_dana', '+1 curated');",
        "INSERT INTO users VALUES ('rdr_extra', '+1');",
    )

    assert missing_seed_rows(world, source) == {
        "users": ("rider_id", ["rdr_o'neil"], ["db/seed.sql"]),
        "bookings": ("booking_ref", ["UB7F3A2C91"], ["db/seed.sql"]),
        "trips": ("", ["2 row(s)"], ["db/seed.sql"]),
    }

    world.executescript(
        "INSERT INTO users VALUES ('rdr_o''neil', '+1');"
        "INSERT INTO bookings VALUES ('UB7F3A2C91', 'matched');"
        "INSERT INTO trips VALUES ('a different trip', 1);"
    )
    assert missing_seed_rows(world, source) == {}


def test_only_seed_rows_tells_seeded_starting_rows_from_leftovers(tmp_path):
    source = _source(tmp_path)
    world = _world("INSERT INTO bookings VALUES ('UB7F3A2C91', 'matched');")

    assert only_seed_rows(world, source, "bookings")
    world.execute("INSERT INTO bookings VALUES ('BUILDER_TEST', 'matched')")
    assert not only_seed_rows(world, source, "bookings")
    assert not only_seed_rows(world, source, "trips")


# A key above 2**53 that a float would round to its neighbour.
_BIG = 9007199254740993


def test_integer_literals_are_read_exactly_and_other_numbers_as_before(tmp_path):
    source = _source(
        tmp_path,
        "INSERT INTO drivers (id) VALUES "
        f"({_BIG}), (-42), (+7), (38.00), (1.5e3), (-2.5E-1), (4.9), "
        "(99999999999999999999);\n",
    )

    assert [row["id"] for _file, row in seed_rows(source)["drivers"]] == [
        str(_BIG),
        "-42",
        "7",
        "38",
        "1500",
        "-0.25",
        "4.9",
        # Past 64 bits SQLite stores a REAL, so the key is read as one.
        "100000000000000000000",
    ]


def test_a_big_integer_key_the_world_holds_is_not_reported_missing(tmp_path):
    source = _source(tmp_path, f"INSERT INTO drivers (id) VALUES ({_BIG});\n")
    world = sqlite3.connect(":memory:")
    world.execute("CREATE TABLE drivers (id INTEGER PRIMARY KEY)")
    world.execute(f"INSERT INTO drivers VALUES ({_BIG - 1})")

    assert missing_seed_rows(world, source) == {
        "drivers": ("id", [str(_BIG)], ["db/seed.sql"])
    }
    world.execute(f"INSERT INTO drivers VALUES ({_BIG})")
    world.execute(f"DELETE FROM drivers WHERE id = {_BIG - 1}")
    assert missing_seed_rows(world, source) == {}
    assert only_seed_rows(world, source, "drivers")


@pytest.mark.parametrize(
    ("key_type", "seeded"), [("TEXT", "'UB7F3A2C91'"), ("BIGINT", str(_BIG))]
)
def test_a_seeded_in_flight_row_is_not_refused_as_left_over_from_building(
    tmp_path, key_type, seeded
):
    from fi.alk.harness.catalogue import Catalogue, SubGoal, save_catalogue
    from fi.alk.harness.contract import AgentContract, ToolSpec
    from fi.alk.harness.simulator import save_simulator_prompt
    from fi.alk.harness.world import GeneratedWorld
    from fi.alk.harness.world.snapshot import save
    from fi.alk.harness.world.tools import world_tools

    class Rides(GeneratedWorld):
        name = "rides"
        tools = [{"name": "book"}]
        handlers = {
            "book": (
                "def handle(args, db):\n"
                "    if args.get('ref') != 'UB_NEW': raise ToolError('no such ride')\n"
                "    db.execute('INSERT INTO bookings VALUES (?, ?)', "
                "[args['ref'], 'matched'])\n"
                "    return {'ok': 1}\n"
            )
        }

    world = Rides(":memory:")
    world.connection.executescript(
        f"CREATE TABLE bookings (booking_ref {key_type} PRIMARY KEY, status TEXT);"
        f"INSERT INTO bookings VALUES ({seeded}, 'matched');"
    )
    world.connection.commit()
    workspace = tmp_path / "work"
    save(
        world,
        workspace,
        notes="rides",
        sequences=[
            {
                "name": "book-one",
                "calls": [{"tool": "book", "arguments": {"ref": "UB_NEW"}}],
                "expect_state": {"bookings.count": 2},
            }
        ],
        world_checks={
            "ride_seeded": "def check(world):\n"
            "    refs = [r['booking_ref'] for r in world.state()['bookings']]\n"
            f"    return None if {seeded} in refs else 'no seeded ride'\n"
        },
    )
    contract = AgentContract(
        agent="rides",
        conversational=True,
        real_use_cases=["book a ride"],
        tools=[ToolSpec(name="book", args=["ref"], arg_values={"ref": ["UB_NEW"]})],
    )
    save_catalogue(
        Catalogue(
            sub_goals=[
                SubGoal(
                    name="booked",
                    what="a ride was booked",
                    check="def check(world, calls):\n"
                    "    return None if any(c.name == 'book' for c in calls) else 'no'\n",
                )
            ]
        ),
        workspace,
    )
    save_simulator_prompt(
        "You are a caller with persona {{ persona }}. Follow {{ instruction }} and ask natural "
        "follow-up questions. Do not act as the assistant or invent its responses.",
        workspace,
    )

    def saved(source_root: str) -> dict:
        server, _world = world_tools(contract, workspace, source_root=source_root)
        handler = next(one.handler for one in server.tools if one.name == "save_world")
        return asyncio.run(handler({}))

    refused = saved("")
    assert refused.get("is_error") and "left over from building" in str(refused)

    source = tmp_path / "source"
    (source / "db").mkdir(parents=True)
    (source / "db" / "seed.sql").write_text(
        f"INSERT INTO bookings (booking_ref, status) VALUES ({seeded}, 'matched');\n",
        encoding="utf-8",
    )
    kept = saved(str(source))
    assert "left over from building" not in str(kept), kept
    assert not kept.get("is_error"), kept
