# Changing the world, and reading it back

Consult this while writing `setup_code`, `ready_code` and any check. None of it names what the
world is kept in, which is deliberate: the store varies more between agents than anything else.

## Changing it directly

For a native file task, use a named workspace declared in `runtime.workspaces`.
Each declaration maps a name to a directory relative to the target process's working directory.
The directory must come from the target's actual configuration. A database collection called
`files` does not populate that directory.

```python
def setup(world):
    world.write_file("project", "notes.txt", "Review the delivery date.\n")

def ready(world):
    if world.read_file("project", "notes.txt") != "Review the delivery date.\n":
        return "The target workspace is missing the expected note."
```

`read_file` returns UTF-8 text and is available in checks. `write_file` is available only in
setup. Both require a declared workspace and a relative path inside it. Missing files, links,
special files and oversized contents raise errors. Reset restores the private runtime copy;
fixtures never modify the submitted source. Use this surface only when the target reads the
same configured directory.

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


## Native Compose runtime

A native Compose world has no harness database or local tool replay. Its state lives in the
submitted services. During authoring, scenario syntax and declarations are checked; readiness,
reference solvability and no-op proof are **DEFERRED**, never reported as passing locally.
Runtime validation must execute setup, ready and a real chat request before scenarios run.

Use `world.request(capability, method, path, json=None, headers=None)` to access a declared HTTP
capability (`target_http` for the native chat service). The result is a dictionary with `status`, `text` and `json` (null for non-JSON).
Paths start with `/` and stay on that capability's origin. Redirects are returned without being
followed. Requests are bounded to 64 KiB, responses to 1 MiB, headers to 16 KiB and network inactivity
to 10 seconds; the scenario phase also has its own deadline. No environment secrets, proxy settings or automatic authentication are exposed.
A token obtained from the application's own login API may be supplied explicitly in headers.
Transport-routing headers, including Host and Proxy-Authorization, are rejected.

Setup may POST, PUT, PATCH or DELETE. Ready and outcome checks may only GET or HEAD.
Setup must create source-supported fixtures and ready must verify the same actual records.
A 2xx status alone is not evidence that the requested data exists. Check exact response values.

Fixture values sent to the simulator must be declared in advance. Setup return values do not
update simulator slots. APIs that require opaque generated IDs or externally supplied login
secrets need a supported binding before those fixtures can be used. Do not invent such values,
seed a shadow store or replace a missing API. Deterministic checks can inspect actual native
tool calls and API responses. Runtime tool result capture is a separate declared seam.
