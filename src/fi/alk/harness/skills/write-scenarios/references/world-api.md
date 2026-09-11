# Changing the world, and reading it back

Consult this while writing `setup_code`, `ready_code` and any check. None of it names what the
world is kept in, which is deliberate: the store varies more between agents than anything else.

## Changing it directly

Where no tool of the agent's can produce the state you need, change the world's collections and
records yourself:

```python
world.put(collection, record)  # add one record; a stored collection owns its own identifier
world.change(collection, key, changes, by=...)  # change one record
world.drop(collection, key, by=...)  # remove one, or all of them with no key
```

Only use `world.put(..., key=...)` for a collection the agent keeps in memory and addresses by an
outside key. A collection held in a store already carries its own identifier in the record, so passing
`key=` there is wrong. `world.state()` shows every collection and what is in it.

Nothing here names the engine underneath, and nothing you write should. The same three calls serve a
world backed by a relational engine, a columnar one, or the agent's own in-process data, because the
harness stands the engine up and the world presents collections and records whatever it is. A setup that
reaches past these calls, to SQL or to any engine's own client, only works for the one world it was
written against.

```python
def setup(world):
    world.change("stock", "widget", {"quantity": 5}, by="item_id")
```

Use the direct route only for states no tool can produce: a record already in a condition the agent
could never create itself.

One exception. Where the contract says the target's store is hardcoded and process-local, with no
configuration seam, `setup_code` cannot alter target records, because the world and the live target
are separate copies. Use only records already in the frozen base, keep setup empty for them, and
settle outcomes from the captured calls. If coverage needs state the base lacks, report that the
target needs a seed or reset seam rather than writing a scenario that cannot run.

### A record needs every field the store requires

Copy the shape of a record already there, timestamps and all. Omitting a required column, or setting it to
nothing, fails the insert:

```
NotNullViolation: null value in column "taken_at" of relation "trips" violates not-null constraint
```

`inspect_world` on that collection shows what a complete record looks like. A nullable field can be left
out; a required one cannot, and the only way to tell is to look.

### Types are the column's, not Python's convenience

A boolean column takes `True` or `False`. Writing `1` fails outright on a store with real booleans:

```
DatatypeMismatch: column "phone_verified" is of type boolean but expression is of type smallint
```

The scenario then dies in `setup_code`, before any conversation, and reports as a setup crash rather than
anything about the agent. Read-back values may print as `1` and `0`, which is display, not type.

### A collection is not always a list

`world.state()` gives every collection this world has, and their shapes differ by agent. A collection
held in a store gives a list of records. One the agent's own code keeps is often a mapping keyed by
identifier, and iterating that yields the keys, which are strings, so reading a field off one fails.

```python
held = world.state()["some_collection"]
records = list(held.values()) if isinstance(held, dict) else held
```

Look before you write. `inspect_world` shows which is which, and this applies to `setup_code`,
`ready_code` and every check.

### ready_code

Python defining `ready(world)`. Return `None` when the world holds what the scenario presumes, or a
sentence naming what is missing. Check the thing your scenario depends on, not everything.

```python
def ready(world):
    rows = world.state()["stock"]
    widget = next((r for r in rows if r["item_id"] == "widget"), None)
    if widget is None:
        return "no widget in stock at all; this scenario is about its last five"
    if widget["quantity"] != 5:
        return f"stock says {widget['quantity']} widgets, this scenario needs exactly 5"
    return None
```
