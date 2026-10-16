"""The startup hook that stamps the simulation mark on every OpenTelemetry resource
(`simulation_mark_bootstrap.py`).

The hook runs inside the CUSTOMER's interpreter, from a `sitecustomize.py` beside a byte copy of
the module, before the customer's first line. Almost everything it does is about the order in which
interpreters, imports and `sys.path` entries happen, so those cases run as real subprocesses; only
the pure parsing runs in this process. Nothing here touches a database or the network.
"""

from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

from fi.alk.harness import process_runtime as pr

_HARNESS = Path(pr.__file__).parent
HOOK = _HARNESS / "simulation_mark_bootstrap.py"

MARK = (
    "project_name=Our%20Agent,fi.simulation.origin=simulation,"
    "fi.simulation.job_id=job-1,fi.simulation.run_id=run-1"
)
# The customer's own pair first, as the sandbox's environment merge leaves it.
VARIABLE = "service.name=cust," + MARK
OURS = {
    "project_name": "Our Agent",
    "fi.simulation.origin": "simulation",
    "fi.simulation.job_id": "job-1",
    "fi.simulation.run_id": "run-1",
}
APPLIED = "simulation mark hook: applied\n"
FAILED = "simulation mark hook: failed\n"

# A stand-in for the SDK's module, for the cases the real constructor cannot be made to produce
# (it raises on demand, it keeps what it was handed).
STUB_RESOURCES = """\
class Resource:
    def __init__(self, attributes, schema_url=None):
        if isinstance(attributes, dict) and attributes.get("boom"):
            raise ValueError("original raised")
        self.attributes = attributes
        self.schema_url = schema_url
"""

# A `Resource` that is a builtin type: its `__init__` cannot be replaced, so the patch cannot land.
STUB_IMMUTABLE_RESOURCES = "Resource = dict\n"

MERGE_SCRIPT = """
import json
from opentelemetry.sdk.resources import Resource
created = Resource.create({"project_name": "prod"})
constructed = Resource(attributes={"project_name": "prod"})
positional = Resource({"k": "v"}, "https://schema.example/1")
nothing = Resource(attributes=None)
print(json.dumps({
    "created": dict(created.attributes),
    "constructed": dict(constructed.attributes),
    "positional": dict(positional.attributes),
    "schema_url": positional.schema_url,
    "nothing": dict(nothing.attributes),
}))
"""

DORMANT_SCRIPT = """
import json, sys
from opentelemetry.sdk.resources import Resource
print(json.dumps({
    "attributes": dict(Resource(attributes={"project_name": "theirs"}).attributes),
    "finders": [type(finder).__name__ for finder in sys.meta_path],
}))
"""

IMPORTED_BEFORE_SCRIPT = """
import json
from opentelemetry.sdk.resources import Resource
import simulation_mark_bootstrap
before = dict(Resource(attributes={"a": "b"}).attributes)
simulation_mark_bootstrap.install()
after = dict(Resource(attributes={"project_name": "prod"}).attributes)
print(json.dumps({"before": before, "after": after}))
"""

RELOAD_SCRIPT = """
import importlib, json, sys
import opentelemetry.sdk.resources as resources
first = dict(resources.Resource(attributes={"project_name": "prod"}).attributes)
importlib.reload(resources)
second = dict(resources.Resource(attributes={"project_name": "prod"}).attributes)
print(json.dumps({
    "first": first,
    "second": second,
    "first_finder": type(sys.meta_path[0]).__name__,
}))
"""

NO_SDK_SCRIPT = """
import json, sys
print(json.dumps({
    "sdk_modules": sorted(n for n in sys.modules if n.split(".")[0] == "opentelemetry"),
    "finder_installed": any(type(f).__name__ == "_MarkFinder" for f in sys.meta_path),
}))
"""

ORIGINAL_RAISES_SCRIPT = """
import json
from opentelemetry.sdk.resources import Resource
try:
    Resource({"boom": "1"})
    outcome = "returned"
except Exception as exc:
    outcome = [type(exc).__name__, str(exc)]
print(json.dumps({"outcome": outcome}))
"""

MERGE_RAISES_SCRIPT = """
import collections.abc, json
from opentelemetry.sdk.resources import Resource
class Exploding(collections.abc.Mapping):
    def __getitem__(self, key):
        raise RuntimeError("merge failed")
    def __iter__(self):
        raise RuntimeError("merge failed")
    def __len__(self):
        return 1
first = Resource(Exploding())
second = Resource(attributes=Exploding())
print(json.dumps({
    "first": type(first.attributes).__name__,
    "second": type(second.attributes).__name__,
    "fine": dict(Resource({"x": "y"}).attributes),
}))
"""

IMMUTABLE_AFTER_SCRIPT = """
import json
import opentelemetry.sdk.resources as resources
print(json.dumps({"imported": resources.Resource is dict}))
"""

IMMUTABLE_BEFORE_SCRIPT = """
import json
import opentelemetry.sdk.resources as resources
import simulation_mark_bootstrap
simulation_mark_bootstrap.install()
print(json.dumps({"imported": resources.Resource is dict}))
"""

# What the real constructor does with calls it cannot merge into. The outcome of each is the
# type and message it raises, or the attributes it returns, so the hook is compared with no hook.
UNMERGEABLE_CALLS_SCRIPT = """
import json
from opentelemetry.sdk.resources import Resource

def outcome(call):
    try:
        resource = call()
    except Exception as exc:
        return ["raised", type(exc).__name__, str(exc)]
    return ["returned", sorted(dict(resource.attributes).items())]

print(json.dumps({
    "none given": outcome(lambda: Resource()),
    "schema url only": outcome(lambda: Resource(schema_url="https://schema.example/1")),
    "pairs": outcome(lambda: Resource([("a", "b")])),
    "pairs by keyword": outcome(lambda: Resource(attributes=[("a", "b")])),
    "text": outcome(lambda: Resource("ab")),
}))
"""

LEGACY_LOADER_SCRIPT = """
import importlib.machinery, json, sys, types

class Legacy:
    # A loader from before `exec_module`: it can only be asked to load the whole module.
    def load_module(self, name):
        module = types.ModuleType(name)
        module.legacy = True
        sys.modules[name] = module
        return module

class Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "opentelemetry.sdk.resources":
            return importlib.machinery.ModuleSpec(fullname, Legacy())
        return None

# Right behind the hook's finder when there is one, which asks it for the spec.
sys.meta_path.insert(1, Finder())
try:
    import opentelemetry.sdk.resources as resources
    outcome = ["imported", getattr(resources, "legacy", None)]
except Exception as exc:
    outcome = ["raised", type(exc).__name__, str(exc)]
print(json.dumps({"outcome": outcome}))
"""

CHAIN_SCRIPT = """
import json, sys
chained = sys.modules.get("_alk_chained_sitecustomize")
print(json.dumps({
    "sitecustomize": sys.modules["sitecustomize"].__file__,
    "chained": chained.__file__ if chained is not None else None,
}))
"""


def _world(root: Path, name: str = "world", *, sitecustomize: bool = True) -> Path:
    """A world directory as the sandbox writes it: a byte copy of the module, and (unless
    `sitecustomize` is false) the small `sitecustomize.py` that installs it."""
    world = root / name
    world.mkdir()
    shutil.copy2(HOOK, world / "simulation_mark_bootstrap.py")
    if sitecustomize:
        (world / "sitecustomize.py").write_text(
            pr._SIMULATION_MARK_SITECUSTOMIZE, encoding="utf-8"
        )
    return world


def _stub_sdk(
    root: Path, resources: str = STUB_RESOURCES, directory: str = "stub-sdk"
) -> Path:
    package = root / directory / "opentelemetry" / "sdk"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "resources.py").write_text(resources)
    return root / directory


def _run(
    script: str,
    *,
    path: list[Path],
    cwd: Path,
    variable: str | None = VARIABLE,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run `script` in a fresh interpreter whose `PYTHONPATH` is `path`, in an environment with
    nothing of the host's OpenTelemetry or Python configuration but the one variable under test."""
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("OTEL_", "PYTHON"))
    }
    env["PYTHONPATH"] = os.pathsep.join(str(entry) for entry in path)
    if variable is not None:
        env["OTEL_RESOURCE_ATTRIBUTES"] = variable
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _reported(result: subprocess.CompletedProcess) -> dict[str, Any]:
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.splitlines()[-1])


def _repository_hook(root: Path) -> tuple[Path, Path]:
    """A repository directory whose `sitecustomize.py` logs the module name it ran under, and
    the log it writes to (passed to the process as `REPO_HOOK_LOG`)."""
    repo = root / "repo"
    repo.mkdir()
    (repo / "sitecustomize.py").write_text(
        "import os\n"
        "with open(os.environ['REPO_HOOK_LOG'], 'a') as log:\n"
        "    log.write(__name__ + '\\n')\n"
    )
    return repo, root / "repo-hook.log"


def _broken_repository_hook(root: Path) -> Path:
    broken = root / "broken"
    broken.mkdir()
    (broken / "sitecustomize.py").write_text("raise RuntimeError('repository hook')\n")
    return broken


def _scenario_sdk_after(tmp_path: Path, world: Path) -> None:
    # The SDK imported after the hook: every constructor passes through the patched one, and
    # the customer's own pair in the variable (`service.name`) is not merged by the hook.
    result = _run(MERGE_SCRIPT, path=[world], cwd=tmp_path)
    reported = _reported(result)
    for key, value in OURS.items():
        assert reported["created"][key] == value
    assert reported["constructed"] == OURS
    assert reported["positional"] == {**OURS, "k": "v"}
    assert reported["schema_url"] == "https://schema.example/1"
    assert reported["nothing"] == OURS
    assert result.stderr == APPLIED


def _scenario_dormant(variable: str | None):
    def scenario(tmp_path: Path, world: Path) -> None:
        # The variable absent, empty, or without the origin pair: nothing is touched and
        # nothing is said.
        result = _run(DORMANT_SCRIPT, path=[world], cwd=tmp_path, variable=variable)
        reported = _reported(result)
        assert reported["attributes"] == {"project_name": "theirs"}
        assert "_MarkFinder" not in reported["finders"]
        assert result.stderr == ""

    return scenario


def _scenario_sdk_before(tmp_path: Path, world: Path) -> None:
    # The SDK imported before the hook ran: patched at once.
    lib = _world(tmp_path, "lib", sitecustomize=False)
    result = _run(IMPORTED_BEFORE_SCRIPT, path=[lib], cwd=tmp_path)
    reported = _reported(result)
    assert reported["before"] == {"a": "b"}
    assert reported["after"] == OURS
    assert result.stderr == APPLIED


def _scenario_reload(tmp_path: Path, world: Path) -> None:
    # Imported after: through the finder, which stays installed, so a reload is patched again
    # (a reload builds a new class, so the line is written again for it).
    result = _run(RELOAD_SCRIPT, path=[world], cwd=tmp_path)
    reported = _reported(result)
    assert reported["first"] == OURS
    assert reported["second"] == OURS
    assert reported["first_finder"] == "_MarkFinder"
    assert result.stderr == APPLIED * 2


def _scenario_never_imports_sdk(tmp_path: Path, world: Path) -> None:
    # A process that never imports the SDK: the hook is armed, imports none of it, and says
    # nothing.
    result = _run(NO_SDK_SCRIPT, path=[world], cwd=tmp_path)
    reported = _reported(result)
    assert reported["sdk_modules"] == []
    assert reported["finder_installed"] is True
    assert result.stderr == ""


def _scenario_original_raises(tmp_path: Path, world: Path) -> None:
    # The original constructor raising passes through unchanged: not swallowed, not a `failed`.
    stub = _stub_sdk(tmp_path)
    result = _run(ORIGINAL_RAISES_SCRIPT, path=[world, stub], cwd=tmp_path)
    assert _reported(result)["outcome"] == ["ValueError", "original raised"]
    assert result.stderr == APPLIED


def _scenario_merge_raises(tmp_path: Path, world: Path) -> None:
    # The merge itself raising: the original still builds what the caller asked for, `failed`
    # is written once however often it happens, and later resources are merged as usual.
    stub = _stub_sdk(tmp_path)
    result = _run(MERGE_RAISES_SCRIPT, path=[world, stub], cwd=tmp_path)
    reported = _reported(result)
    assert reported["first"] == "Exploding"
    assert reported["second"] == "Exploding"
    assert reported["fine"] == {"x": "y", **OURS}
    assert result.stderr == APPLIED + FAILED


def _scenario_chain_layouts(tmp_path: Path, world: Path) -> None:
    # A repository's own `sitecustomize.py` later on the path still runs, once, under a private
    # name, and ours stays `sitecustomize`. The world directory may sit on the path again under
    # another spelling (Python drops only a repeat of the same spelling), and that second
    # `sitecustomize.py` is ours, not the next one.
    alias = tmp_path / "world-alias"
    alias.symlink_to(world, target_is_directory=True)
    repo, log = _repository_hook(tmp_path)
    for layout in ([world, repo], [world, alias, repo], [world, repo, alias]):
        log.unlink(missing_ok=True)
        result = _run(
            CHAIN_SCRIPT,
            path=layout,
            cwd=tmp_path,
            extra_env={"REPO_HOOK_LOG": str(log)},
        )
        reported = _reported(result)
        assert (
            Path(reported["sitecustomize"]).resolve()
            == (world / "sitecustomize.py").resolve()
        )
        assert (
            Path(reported["chained"]).resolve() == (repo / "sitecustomize.py").resolve()
        )
        assert log.read_text().splitlines() == ["_alk_chained_sitecustomize"]
        assert result.stderr == ""


def _scenario_chain_when_dormant(tmp_path: Path, world: Path) -> None:
    # The chain is reached when the hook returns early, too.
    repo, log = _repository_hook(tmp_path)
    result = _run(
        CHAIN_SCRIPT,
        path=[world, repo],
        cwd=tmp_path,
        variable=None,
        extra_env={"REPO_HOOK_LOG": str(log)},
    )
    _reported(result)
    assert log.read_text().splitlines() == ["_alk_chained_sitecustomize"]


def _scenario_broken_repo_hook(tmp_path: Path, world: Path) -> None:
    # A repository hook that fails neither stops the process nor costs the mark, and is not
    # reported as a failure of ours: Python reports it, in its own words, exactly as it does
    # when no hook of ours sits in front of it.
    broken = _broken_repository_hook(tmp_path)
    without = _run(NO_SDK_SCRIPT, path=[broken], cwd=tmp_path, variable=None)
    _reported(without)
    # The baseline does report it, so the comparison below is not between two empty strings.
    assert "repository hook" in without.stderr
    with_hook = _run(NO_SDK_SCRIPT, path=[world, broken], cwd=tmp_path)
    _reported(with_hook)
    assert with_hook.stderr == without.stderr
    result = _run(MERGE_SCRIPT, path=[world, broken], cwd=tmp_path)
    assert _reported(result)["constructed"] == OURS
    assert result.stderr == without.stderr + APPLIED


def _scenario_verbose(tmp_path: Path, world: Path) -> None:
    # With verbose startup Python prints the traceback instead, through the same excepthook.
    broken = _broken_repository_hook(tmp_path)
    verbose = {"PYTHONVERBOSE": "1"}
    without = _run(
        NO_SDK_SCRIPT, path=[broken], cwd=tmp_path, variable=None, extra_env=verbose
    )
    with_hook = _run(
        NO_SDK_SCRIPT, path=[world, broken], cwd=tmp_path, extra_env=verbose
    )
    for printed in (without, with_hook):
        _reported(printed)
        assert "Traceback (most recent call last)" in printed.stderr
        assert "RuntimeError: repository hook" in printed.stderr
        assert "set PYTHONVERBOSE for traceback" not in printed.stderr


def _scenario_system_exit(tmp_path: Path, world: Path) -> None:
    # Only an `Exception` is reported and survived. `SystemExit` ends the start-up as it does
    # without the hook.
    leaving = tmp_path / "leaving"
    leaving.mkdir()
    (leaving / "sitecustomize.py").write_text("raise SystemExit(3)\n")
    without = _run(NO_SDK_SCRIPT, path=[leaving], cwd=tmp_path, variable=None)
    with_hook = _run(NO_SDK_SCRIPT, path=[world, leaving], cwd=tmp_path)
    assert without.returncode != 0
    assert with_hook.returncode == without.returncode
    assert with_hook.stdout == without.stdout == ""


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(_scenario_sdk_after, id="sdk_after"),
        pytest.param(_scenario_sdk_before, id="sdk_before"),
        pytest.param(_scenario_reload, id="reload"),
        pytest.param(_scenario_original_raises, id="original_raises"),
        pytest.param(_scenario_merge_raises, id="merge_raises"),
    ],
)
def test_hook_merges_the_mark_into_every_resource(tmp_path: Path, scenario) -> None:
    scenario(tmp_path, _world(tmp_path))


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(_scenario_dormant(None), id="absent"),
        pytest.param(_scenario_dormant(""), id="empty"),
        pytest.param(
            _scenario_dormant(
                "project_name=Our%20Agent,fi.simulation.run_id=run-1,service.name=c"
            ),
            id="no_origin",
        ),
        pytest.param(_scenario_never_imports_sdk, id="never_imports_sdk"),
    ],
)
def test_hook_touches_nothing_without_the_origin_pair_or_the_sdk(
    tmp_path: Path, scenario
) -> None:
    scenario(tmp_path, _world(tmp_path))


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(_scenario_chain_layouts, id="chain_layouts"),
        pytest.param(_scenario_chain_when_dormant, id="chain_when_dormant"),
        pytest.param(_scenario_broken_repo_hook, id="broken_repo_hook"),
        pytest.param(_scenario_verbose, id="verbose"),
        pytest.param(_scenario_system_exit, id="system_exit"),
    ],
)
def test_hook_hands_on_to_the_next_sitecustomize(tmp_path: Path, scenario) -> None:
    scenario(tmp_path, _world(tmp_path))


def test_hook_parses_the_variable_as_the_sdk_does_and_keeps_only_ours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location("mark_hook_under_test", HOOK)
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)

    monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
    assert hook._read_pairs() is None
    for variable in ("", "project_name=x", "fi.simulation.run_id=r,service.name=c"):
        monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", variable)
        assert hook._read_pairs() is None, variable

    # Split on commas, each item on its first `=`, both sides stripped, the value
    # percent-decoded, an item without `=` skipped, the later pair winning a repeated key, and
    # nothing kept but `project_name` and the `fi.simulation.` pairs.
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        " project_name = My%20Agent ,service.name=cust,fi.simulation.origin=simulation,"
        "fi.simulation.run_id= r%2F1 ,junk,fi.simulation.job_id=a=b,"
        "fi.simulation.run_id=r2,=empty,deployment.environment=prod",
    )
    assert hook._read_pairs() == {
        "project_name": "My Agent",
        "fi.simulation.origin": "simulation",
        "fi.simulation.run_id": "r2",
        "fi.simulation.job_id": "a=b",
    }


def test_hook_patches_an_sdk_imported_from_a_zip_and_tests_the_module_name(
    tmp_path: Path,
) -> None:
    """One `zipimporter` serves every module of its archive, so its wrapped `exec_module` also
    runs for modules that are not the SDK's resources and must leave them alone."""
    archive = tmp_path / "sdk.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("opentelemetry/__init__.py", "")
        bundle.writestr("opentelemetry/sdk/__init__.py", "")
        bundle.writestr("opentelemetry/sdk/resources.py", STUB_RESOURCES)
        bundle.writestr("opentelemetry/sdk/other.py", "VALUE = 7\n")
    script = """
import json
from opentelemetry.sdk import resources
import opentelemetry.sdk.other as other
print(json.dumps({
    "loader": type(resources.__loader__).__name__,
    "attributes": dict(resources.Resource({"x": "y"}).attributes),
    "other": other.VALUE,
}))
"""
    result = _run(script, path=[_world(tmp_path), archive], cwd=tmp_path)
    reported = _reported(result)
    assert reported["loader"] == "zipimporter"
    assert reported["attributes"] == {"x": "y", **OURS}
    assert reported["other"] == 7
    assert result.stderr == APPLIED


def test_finder_never_raises_out_of_find_spec(tmp_path: Path) -> None:
    script = """
import importlib.machinery, json, sys

finder = sys.meta_path[0]

class Raising:
    def find_spec(self, fullname, path=None, target=None):
        raise RuntimeError("someone else's finder")

class Refusing:
    __slots__ = ()
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        pass

offered = importlib.machinery.ModuleSpec("opentelemetry.sdk.resources", Refusing())

class Offering:
    def find_spec(self, fullname, path=None, target=None):
        return offered

raising = Raising()
sys.meta_path.insert(1, raising)
other_name = finder.find_spec("json.tool", None)
raised_around = finder.find_spec("opentelemetry.sdk.resources", None)
sys.meta_path.remove(raising)
sys.meta_path.insert(1, Offering())
returned = finder.find_spec("opentelemetry.sdk.resources", None)
print(json.dumps({
    "other_name": other_name,
    "raised_around": raised_around,
    "offered_returned": returned is offered,
}))
"""
    result = _run(script, path=[_world(tmp_path)], cwd=tmp_path)
    reported = _reported(result)
    # Another name is none of its business; another finder's error is that finder's to report;
    # a loader it cannot wrap still gets its spec back, and the hook says it failed.
    assert reported == {
        "other_name": None,
        "raised_around": None,
        "offered_returned": True,
    }
    assert result.stderr == FAILED


def test_an_sdk_whose_constructor_cannot_be_patched_still_imports_and_the_hook_says_so(
    tmp_path: Path,
) -> None:
    """The customer's own import of the SDK must never be the thing that fails. A builtin
    `Resource` has an `__init__` that cannot be assigned, which stands for any class shape the
    patch does not expect."""
    stub = _stub_sdk(tmp_path, STUB_IMMUTABLE_RESOURCES, "stub-immutable")

    # Imported after the hook: the patch runs inside the import itself.
    result = _run(IMMUTABLE_AFTER_SCRIPT, path=[_world(tmp_path), stub], cwd=tmp_path)
    assert _reported(result) == {"imported": True}
    assert result.stderr == FAILED

    # Imported before the hook: the patch runs inside `install()`, which the customer's
    # interpreter calls at start-up.
    lib = _world(tmp_path, "lib", sitecustomize=False)
    result = _run(IMMUTABLE_BEFORE_SCRIPT, path=[lib, stub], cwd=tmp_path)
    assert _reported(result) == {"imported": True}
    assert result.stderr == FAILED


def test_calls_the_constructor_cannot_merge_into_behave_as_without_the_hook(
    tmp_path: Path,
) -> None:
    bare = _world(tmp_path, "bare", sitecustomize=False)
    without = _reported(
        _run(UNMERGEABLE_CALLS_SCRIPT, path=[bare], cwd=tmp_path, variable=None)
    )
    # The comparison means something only if the SDK itself turns these calls down.
    assert without["none given"][0] == "raised"

    result = _run(UNMERGEABLE_CALLS_SCRIPT, path=[_world(tmp_path)], cwd=tmp_path)
    assert _reported(result) == without
    assert result.stderr == APPLIED


def test_a_loader_without_exec_module_loads_as_without_the_hook_and_is_reported(
    tmp_path: Path,
) -> None:
    bare = _world(tmp_path, "bare", sitecustomize=False)
    without = _run(LEGACY_LOADER_SCRIPT, path=[bare], cwd=tmp_path, variable=None)
    outcome = _reported(without)["outcome"]
    assert without.stderr == ""

    # The patch cannot land through such a loader, so the SDK loads unmarked and the log says
    # so; the customer's import is no different from what it would be without the hook.
    result = _run(LEGACY_LOADER_SCRIPT, path=[_world(tmp_path)], cwd=tmp_path)
    assert _reported(result)["outcome"] == outcome
    assert result.stderr == FAILED


def test_a_second_install_or_a_second_copy_of_the_module_does_not_wrap_twice(
    tmp_path: Path,
) -> None:
    world = _world(tmp_path)
    script = """
import importlib, importlib.util, json, os, sys
from opentelemetry.sdk.resources import Resource
import opentelemetry.sdk.resources as resources
import simulation_mark_bootstrap

simulation_mark_bootstrap.install()
simulation_mark_bootstrap.install()
finders = sum(
    type(finder) is simulation_mark_bootstrap._MarkFinder for finder in sys.meta_path
)
spec = importlib.util.spec_from_file_location(
    "second_copy", os.path.join(os.environ["WORLD"], "simulation_mark_bootstrap.py")
)
second = importlib.util.module_from_spec(spec)
spec.loader.exec_module(second)
second.install()
importlib.reload(resources)
print(json.dumps({
    "attributes": dict(resources.Resource({"x": "y"}).attributes),
    "finders": finders,
}))
"""
    result = _run(script, path=[world], cwd=tmp_path, extra_env={"WORLD": str(world)})
    reported = _reported(result)
    assert reported["attributes"] == {"x": "y", **OURS}
    # The sitecustomize already installed it, so neither call here adds a finder.
    assert reported["finders"] == 1
    # One line for the first patch and one for the class the reload made; the second install
    # and the second copy patched nothing the first had not.
    assert result.stderr == APPLIED * 2


def test_two_copies_of_the_finder_never_call_each_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Loader:
        def exec_module(self, module: Any) -> None:
            pass

    class Stub:
        def __init__(self) -> None:
            self.calls = 0

        def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
            if fullname != "opentelemetry.sdk.resources":
                return None
            self.calls += 1
            return importlib.machinery.ModuleSpec(fullname, Loader())

    def finder_of_a_fresh_copy(name: str) -> Any:
        spec = importlib.util.spec_from_file_location(name, HOOK)
        copy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(copy)
        copy._PATCHES["opentelemetry.sdk.resources"] = lambda module: None
        return copy._MarkFinder()

    first = finder_of_a_fresh_copy("mark_hook_copy_one")
    second = finder_of_a_fresh_copy("mark_hook_copy_two")
    stub = Stub()
    # Each copy sits ahead of the real finders. A copy that delegated to the other would be
    # delegated back to, level after level, until the interpreter's limit raised and the
    # handler swallowed it: the import still succeeds, but every level asks the real finders
    # again, so the count is the only place the loop shows.
    monkeypatch.setattr(sys, "meta_path", [first, second, stub, *sys.meta_path])

    spec = first.find_spec("opentelemetry.sdk.resources", None)

    assert spec is not None
    assert stub.calls == 1


def test_hook_module_imports_only_the_standard_library_and_uses_python_39_syntax() -> (
    None
):
    """It runs inside the customer's interpreter, which has no `fi` and may be older than the
    package's own floor."""
    tree = ast.parse(HOOK.read_text(encoding="utf-8"), feature_version=(3, 9))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imported.add(node.module or "")
    assert imported == {
        "__future__",
        "collections.abc",
        "functools",
        "importlib.machinery",
        "importlib.util",
        "inspect",
        "os",
        "sys",
        "urllib.parse",
    }


COLLECTOR = "http://collector.test:4318"
INGEST = {
    "FI_SIMULATION_INGEST_API_KEY": "ingest-key",
    "FI_SIMULATION_INGEST_SECRET_KEY": "ingest-secret",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": f"{COLLECTOR}/v1/traces",
}
CUSTOMER_KEYS = {"FI_API_KEY": "customer-key", "FI_SECRET_KEY": "customer-secret"}
OUR_AUTH = {"x-api-key": "ingest-key", "x-secret-key": "ingest-secret"}

EXPORTER_HEADERS = """
import json, os, sys
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as Http
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as Grpc

def sent(exporter):
    return {name.lower(): value for name, value in dict(exporter._headers).items()}

def sent_exact(exporter):
    return dict(exporter._headers)
"""


def _export_run(
    tmp_path: Path, script: str, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    return _run(
        EXPORTER_HEADERS + script,
        path=[_world(tmp_path)],
        cwd=tmp_path,
        extra_env={**INGEST, **CUSTOMER_KEYS, **(extra_env or {})},
    )


def test_declared_fi_key_stays_in_the_environment_and_the_exporter_sends_with_the_ingest_key(
    tmp_path: Path,
) -> None:
    script = """
from fi_instrumentation.otel import GRPCSpanExporter, HTTPSpanExporter

print(json.dumps({
    "http": sent(Http(endpoint=os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"])),
    "grpc": sent(Grpc(endpoint="collector.test:4318", insecure=True)),
    "traceai_http": sent(HTTPSpanExporter()),
    "traceai_grpc": sent(GRPCSpanExporter()),
    "keys": [os.environ.get("FI_API_KEY"), os.environ.get("FI_SECRET_KEY")],
}))
"""
    reported = _reported(
        _export_run(
            tmp_path,
            script,
            {"FI_BASE_URL": COLLECTOR, "FI_GRPC_URL": "elsewhere.test:4317"},
        )
    )

    for name in ("http", "grpc", "traceai_http"):
        assert {key: reported[name].get(key) for key in OUR_AUTH} == OUR_AUTH, name
    assert reported["keys"] == ["customer-key", "customer-secret"]
    assert reported["traceai_grpc"]["x-api-key"] == "customer-key"


def test_explicit_headers_in_code_keep_their_other_entries_and_lose_only_fi_auth(
    tmp_path: Path,
) -> None:
    script = """
endpoint = os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]
print(json.dumps({
    "dict": sent(Http(endpoint=endpoint, headers={"X-Api-Key": "c", "X-Team": "a"})),
    "pairs": sent(Grpc(
        endpoint="collector.test:4318",
        insecure=True,
        headers=[("x-secret-key", "c"), ("x-team", "a")],
    )),
    "text": sent(Grpc(
        endpoint="collector.test:4318", insecure=True, headers="X-API-KEY=c,x-team=a"
    )),
}))
"""
    reported = _reported(_export_run(tmp_path, script))

    for form in ("dict", "pairs", "text"):
        assert reported[form] == {"x-team": "a", **OUR_AUTH}, form


def test_exporter_aimed_at_another_host_is_untouched(tmp_path: Path) -> None:
    script = """
theirs = {"x-api-key": "c", "x-team": "a"}
print(json.dumps({
    "host": sent(Http(endpoint="http://elsewhere.test:4318/v1/traces", headers=theirs)),
    "port": sent(Http(endpoint="http://collector.test:9999/v1/traces", headers=theirs)),
    "grpc": sent(Grpc(endpoint="elsewhere.test:4317", insecure=True, headers=theirs)),
}))
"""
    reported = _reported(_export_run(tmp_path, script))

    for case in ("host", "port", "grpc"):
        assert reported[case] == {"x-api-key": "c", "x-team": "a"}, case


@pytest.mark.parametrize(
    "before",
    [
        {"OTEL_EXPORTER_OTLP_TRACES_HEADERS": "x-team=a,X-Api-Key=c"},
        {"OTEL_EXPORTER_OTLP_HEADERS": "x-team=b,x-secret-key=c"},
        {},
    ],
    ids=["traces_value", "generic_value_only", "neither"],
)
def test_the_headers_variables_are_never_written(
    tmp_path: Path, before: dict[str, str]
) -> None:
    script = """
print(json.dumps({
    "traces": os.environ.get("OTEL_EXPORTER_OTLP_TRACES_HEADERS"),
    "generic": os.environ.get("OTEL_EXPORTER_OTLP_HEADERS"),
}))
"""
    reported = _reported(_export_run(tmp_path, script, before))

    assert reported["traces"] == before.get("OTEL_EXPORTER_OTLP_TRACES_HEADERS")
    assert reported["generic"] == before.get("OTEL_EXPORTER_OTLP_HEADERS")


def test_an_exporter_built_without_headers_for_another_vendor_gets_no_ingest_key(
    tmp_path: Path,
) -> None:
    script = """
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "x-team=a"
print(json.dumps({
    "http": sent(Http(endpoint="http://elsewhere.test:4318/v1/traces")),
    "grpc": sent(Grpc(endpoint="elsewhere.test:4317", insecure=True)),
    "traces": os.environ.get("OTEL_EXPORTER_OTLP_TRACES_HEADERS"),
}))
"""
    reported = _reported(_export_run(tmp_path, script))

    assert reported["http"] == {"x-team": "a"}
    assert reported["grpc"] == {"x-team": "a"}
    assert reported["traces"] is None


def test_grpc_gets_lower_case_names_and_http_keeps_the_names_it_was_given(
    tmp_path: Path,
) -> None:
    script = """
endpoint = os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]
print(json.dumps({
    "grpc": sent_exact(Grpc(endpoint="collector.test:4318", insecure=True)),
    "http": sent_exact(Http(endpoint=endpoint, headers={"X-Team": "a"})),
}))
"""
    reported = _reported(_export_run(tmp_path, script))

    assert set(reported["grpc"]) == set(OUR_AUTH)
    assert reported["http"] == {"X-Team": "a", **OUR_AUTH}


def test_an_exporter_built_with_no_endpoint_argument_reads_the_environment(
    tmp_path: Path,
) -> None:
    script = """
print(json.dumps({"http": sent(Http()), "grpc": sent(Grpc(insecure=True))}))
"""
    extra = {
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://collector.test:4318/v1/traces"
    }
    reported = _reported(_export_run(tmp_path, script, extra))

    assert {key: reported["http"].get(key) for key in OUR_AUTH} == OUR_AUTH


def test_traces_headers_win_over_the_generic_ones_and_ours_win_over_both(
    tmp_path: Path,
) -> None:
    script = """
print(json.dumps(sent(Http(endpoint=os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]))))
"""
    before = {
        "OTEL_EXPORTER_OTLP_HEADERS": "x-team=generic,x-only-generic=1,x-api-key=g",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS": "x-team=traces,X-Secret-Key=t",
    }
    reported = _reported(_export_run(tmp_path, script, before))

    assert reported == {"x-team": "traces", "x-only-generic": "1", **OUR_AUTH}


def test_a_generic_header_written_after_install_reaches_an_exporter_built_without_headers(
    tmp_path: Path,
) -> None:
    script = """
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "x-late=1,x-api-key=c"
print(json.dumps(sent(Http(endpoint=os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]))))
"""
    reported = _reported(_export_run(tmp_path, script))

    assert reported == {"x-late": "1", **OUR_AUTH}


def test_exporter_module_imported_before_the_hook_is_patched_at_once(
    tmp_path: Path,
) -> None:
    lib = _world(tmp_path, "lib", sitecustomize=False)
    script = """
import json, os
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as Http
import simulation_mark_bootstrap

simulation_mark_bootstrap.install()
exporter = Http(
    endpoint=os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"], headers={"x-api-key": "c"}
)
print(json.dumps({name.lower(): value for name, value in exporter._headers.items()}))
"""
    result = _run(
        script, path=[lib], cwd=tmp_path, extra_env={**INGEST, **CUSTOMER_KEYS}
    )

    assert _reported(result) == OUR_AUTH
    assert APPLIED in result.stderr


@pytest.mark.parametrize(
    "missing",
    [
        "FI_SIMULATION_INGEST_API_KEY",
        "FI_SIMULATION_INGEST_SECRET_KEY",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    ],
)
def test_no_ingest_variables_no_export_patch(tmp_path: Path, missing: str) -> None:
    script = """
print(json.dumps({
    "headers": sent(Http(endpoint="http://collector.test:4318/v1/traces", headers={"x-api-key": "c"})),
    "patched": getattr(Http.__init__, "_alk_simulation_mark", False),
    "traces": os.environ.get("OTEL_EXPORTER_OTLP_TRACES_HEADERS"),
}))
"""
    environment = {**INGEST, **CUSTOMER_KEYS}
    del environment[missing]
    result = _run(
        EXPORTER_HEADERS + script,
        path=[_world(tmp_path)],
        cwd=tmp_path,
        extra_env=environment,
    )

    assert _reported(result) == {
        "headers": {"x-api-key": "c"},
        "patched": False,
        "traces": None,
    }


def test_wrapper_failure_runs_the_original_constructor(tmp_path: Path) -> None:
    script = """
import simulation_mark_bootstrap

def boom(*args, **kwargs):
    raise RuntimeError("ingest-key")

simulation_mark_bootstrap._with_ingest = boom
exporter = Http(
    endpoint=os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"],
    headers={"x-api-key": "c", "x-team": "a"},
)
print(json.dumps(sent(exporter)))
"""
    result = _export_run(tmp_path, script)

    assert _reported(result) == {"x-api-key": "", "x-secret-key": "", "x-team": "a"}
    assert result.stderr.count(FAILED) == 1


def test_a_failed_patch_never_sends_the_customers_own_key_to_our_collector(
    tmp_path: Path,
) -> None:
    script = """
endpoint = os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]
print(json.dumps({
    "unreadable": sent(Http(endpoint=endpoint, headers=(pair for pair in [("x-api-key", "c"), ("x-team", "a")]))),
    "from_environment": sent(Http(endpoint=endpoint, headers=(p for p in []))),
    "elsewhere": sent(Http(endpoint="http://elsewhere.test:4318/v1/traces", headers={"x-api-key": "c"})),
}))
"""
    reported = _reported(_export_run(tmp_path, script))

    for name in ("unreadable", "from_environment"):
        assert reported[name].get("x-api-key") == "", name
        assert reported[name].get("x-secret-key") == "", name
    assert reported["elsewhere"] == {"x-api-key": "c"}


def test_a_bad_collector_port_does_not_cost_the_process_its_resource_marks(
    tmp_path: Path,
) -> None:
    script = """
from opentelemetry.sdk.resources import Resource

attributes = dict(Resource(attributes={"a": "b"}).attributes)
print(json.dumps({"marked": attributes.get("fi.simulation.origin")}))
"""
    bad = {
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://collector.test:notaport/v1/traces"
    }
    result = _export_run(tmp_path, script, bad)

    assert _reported(result) == {"marked": "simulation"}
    assert result.stderr.count(FAILED) == 1


def test_no_key_in_stderr(tmp_path: Path) -> None:
    script = """
import simulation_mark_bootstrap

endpoint = os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]
Http(endpoint=endpoint)
Grpc(endpoint="collector.test:4318", insecure=True)

def boom(*args, **kwargs):
    raise RuntimeError(os.environ["FI_SIMULATION_INGEST_SECRET_KEY"])

simulation_mark_bootstrap._with_ingest = boom
Http(endpoint=endpoint)
print(json.dumps({}))
"""
    result = _export_run(tmp_path, script)

    _reported(result)
    assert set(result.stderr.splitlines(keepends=True)) == {APPLIED, FAILED}
    for secret in ("ingest-key", "ingest-secret", "customer-key", "customer-secret"):
        assert secret not in result.stderr
