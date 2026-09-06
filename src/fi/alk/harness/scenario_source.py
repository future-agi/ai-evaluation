"""The scenario-source adapter: reads generated scenario documents out of the bundle (code-as-text,
on the on-disk layout `folder.py` documents) and turns them into the `Scenario`/`SubGoal` objects
`hosted_scheduler.py` actually drives.

Deliberately does NOT import `fi.alk.harness.folder` or `fi.alk.harness.scenario` for the model:
both exist at HEAD, but HEAD's `Scenario` carries no `scenario_key`/`scenario_id` (those are
pr63-only) and its default `extra="ignore"` would silently discard exactly the two fields the
scheduler needs off a `scenario.json` written in the newer shape. So this module reads
`scenario.json` as a plain dict and pulls fields out by key, mirroring the documented layout
instead of depending on either model -- see the report's design-decisions section for the
consequences of that choice (HEAD-model drift).

RESOLVED (p13-worker-r2, reports/p13-worker-r2.md CONTRACT NOTES): the `provision`/`begin` wire
shapes below follow the platform's actual, live route (futureagi/simulate/serializers/services/
views `hosted_harness.py`) rather than Karthik's Scenario Generation Contract text (PR #63), where
the two disagree -- a single `POST .../scenarios/` discriminated by a body-level `operation` field,
`begin` keyed on the full `scenario_keys` set, and a provision response KEYED by `scenario_key`
(never a position-ordered array). `register_with_platform` below is the seam that builds those
payloads and merges the platform-assigned `scenario_id`s back onto each scenario.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from . import outbound as ob
from .catalogue import CATALOGUE
from .job import FailureDomain

if TYPE_CHECKING:
    from .hosted_entrypoint import ScenariosClient

# LAYOUT DECISION (contract-silent -- hosted-execution-seams.md v1.15 §2 never mentions scenario
# documents, and §7 assigns the on-disk layout to Karthik's contract, status "in review"). Scenario
# documents live at `<bundle_dir>/<SCENARIOS_DIRNAME>/<name>/...`, matching `folder.py`'s own
# `SCENARIOS` constant, so a write_folder destination of `<bundle_dir>` lands correctly with no
# translation. Kept as one module-level constant so a later contract can move it in one edit.
SCENARIOS_DIRNAME = "scenarios"

_CHECKS_DIRNAME = "checks"
_SCENARIO_JSON = "scenario.json"
_SETUP_PY = "setup.py"
_READY_PY = "ready.py"

# R1-5: divergence (a) (compile once, at load) moves every scenario file's compile+exec into
# `asyncio.to_thread(load_scenarios, ...)`, OUTSIDE every phase budget the scheduler enforces
# (`SETUP_TIMEOUT_SECONDS`/`READY_TIMEOUT_SECONDS`/`CHECK_TIMEOUT_SECONDS` all apply downstream, to
# `_run_phase`). Pathological module-level code (`while True: pass`, a blocking socket read) would
# otherwise hang the load with zero terminal events, no timeout catching it, ever. One generous
# wall-clock budget here converts that hang into a typed terminal instead -- a worker thread cannot
# actually be killed, so this accepts a leaked thread over an unbounded one, on the reasoning that a
# terminal event today is strictly better than none ever.
_LOAD_TIMEOUT_SECONDS = 60.0

# Every scenario a job asked for is called. A job that wrote two hundred is asking for two hundred
# calls: that is the product, and calling fewer would quietly deliver a fraction of what somebody
# paid for. There is deliberately no setting here; a smaller run is a smaller `scenario_count`.
def sampled_for_calling(scenarios: Sequence[Any]) -> list[Any]:
    """Which scenarios this job calls, which is all of them."""
    return list(scenarios)

# The TEXT of a judged sub-goal's check is never persisted by `folder.py`'s `write_folder` (only
# `SubGoal.deterministic()` entries get a `checks/<name>.py` file) -- this fixed marker stands in
# for it so `SubGoal.judged` (mandatory; read by plain attribute access in `hosted_scheduler.py`)
# is never empty for a sub-goal this reader knows is judged. The round-trip loss this represents is
# recorded under CONTRACT QUESTIONS in the report.
_JUDGED_MARKER = "judged (reason not persisted by folder.py's on-disk layout)"


class ScenarioDocumentInvalid(RuntimeError):
    """A scenario folder under `<bundle_dir>/scenarios/` is unreadable or malformed: missing
    `scenario.json`, invalid JSON, a field of the wrong shape, or a `setup.py`/`ready.py`/
    `checks/<goal>.py` that will not compile. Raised rather than skipped -- `folder.py`'s own
    `read_all` swallows exactly this and continues, which is right for a human editing a suite by
    hand and wrong for a hosted job, where a bad scenario silently vanishing from the run reads as
    a suite that passed with fewer scenarios than it should have."""


def bundle_has_scenarios(bundle_dir: Path) -> bool:
    """The LAYOUT DECISION's presence test: `<bundle_dir>/scenarios/` exists and at least one of
    its subdirectories holds a `scenario.json`. Deliberately narrow -- an empty or missing
    `scenarios/` must not flip the wiring decision away from the safe `NotWiredScenarioSource`
    default. A bundle that means to carry scenarios but got the layout wrong fails loudly once
    `load_scenarios` actually reads it, not silently by being treated as scenario-free here.

    An unreadable `scenarios/` directory (permission denied, race with deletion, etc.) reports
    False rather than raising `OSError` (R1-1): this call sits in `hosted_entrypoint.py`'s wiring
    `if` BEFORE the `try`/`except` that maps `ScenarioDocumentInvalid` to a typed terminal, so an
    escape here would kill the whole guest process with no terminal event at all. Falling back to
    `NotWiredScenarioSource`'s existing typed failure is the safe direction -- the alternative of
    raising here has nowhere typed to land.
    """
    root = bundle_dir / SCENARIOS_DIRNAME
    if not root.is_dir():
        return False
    try:
        children = list(root.iterdir())
    except OSError:
        return False
    return any((child / _SCENARIO_JSON).is_file() for child in children if child.is_dir())


def _judged_placeholder_check(world: Any, calls: Any) -> None:
    """The check for a judged (non-deterministic) sub-goal: always reports "held", via the same
    convention a deterministic check uses. `SubGoalResult.judged` -- not this return value -- is
    what has to tell downstream a real judge still has to run; see CONTRACT QUESTIONS in the
    report for the gap that leaves (a judged-only scenario reports "passed" before any judge runs).
    """
    del world, calls
    return None


def _compile_entry(
    source: str, *, label: str, entry: str, allow_empty: bool = True
) -> Callable[..., object]:
    """One scenario code-text -> a bare callable that raises, compiled ONCE here rather than per
    call. Mirrors `folder.py`'s `_run` in exactly two respects: `compile(source, name, "exec")`
    into a fresh, empty-dict namespace with default builtins, and (when `allow_empty`)
    empty/whitespace-only source is a no-op success. Deliberately diverges from `_run` in the two
    respects the brief calls out:
    (a) compiling here, at load, turns a syntax error into one typed terminal for the whole job
    instead of a per-scenario fault discovered mid-run; (b) the compiled function is returned
    BARE, never wrapped in `_run`'s `Outcome` -- `hosted_scheduler.py`'s `_run_phase` is what
    classifies a raised exception into `setup_crashed`/`ready_broken`/`check_broken`/a timeout, and
    an `Outcome` return here would swallow every one of those before `_run_phase` ever saw it.
    `_run`'s complaint-sentence return convention (a non-None, non-True, non-empty-string value
    means "did not hold") is left untranslated for the same reason: `hosted_scheduler.py`'s own
    `_classify_ready`/`_classify_check` already own that classification on the return-VALUE side of
    this boundary; only the raise-vs-return boundary belongs to this module.

    `allow_empty=False` is for `check` entries only (R1-2): an EXISTING `checks/<name>.py` that is
    empty or whitespace-only compiles to nothing, and handing back a no-op "held" callable -- the
    right behavior for "no setup/ready code here" -- would silently turn "there is no check" into
    "the check passed": a vacuous deterministic pass, which is exactly what `hosted_scheduler.py`
    forbids one scenario level up ("scenario declared zero sub_goals"). Absence of the file
    entirely is what means "judged" (see `_load_one`); an existing-but-empty file is malformed.
    """
    if not source.strip():
        if allow_empty:
            return lambda *args: None
        raise ScenarioDocumentInvalid(f"{label} defines no {entry}()")
    try:
        code = compile(source, f"<{label}>", "exec")
    except (SyntaxError, ValueError) as exc:
        # SyntaxError is the common case; a NUL byte in the source raises ValueError on some
        # interpreter versions (R1-1) rather than SyntaxError -- both are the same content defect.
        raise ScenarioDocumentInvalid(f"{label} would not compile: {exc}") from exc
    namespace: dict[str, Any] = {}
    try:
        exec(code, namespace)  # noqa: S102 - scenario code is meant to be exec'd; see CONTRACT QUESTIONS
    except (Exception, SystemExit, KeyboardInterrupt) as exc:  # noqa: BLE001 - see R1-1
        # A module-level `sys.exit()`/`raise SystemExit(...)` in the file itself is a malformed
        # document, not a request to shut the guest process down -- `SystemExit`/`KeyboardInterrupt`
        # are `BaseException`, not `Exception`, so a bare `except Exception` (the pre-R1-1 shape)
        # let them straight through this boundary and out of `run_job` with zero terminal events,
        # the guest exiting with whatever code the scenario file itself chose.
        raise ScenarioDocumentInvalid(f"{label} would not compile: {exc}") from exc
    function = namespace.get(entry)
    if not callable(function):
        raise ScenarioDocumentInvalid(f"{label} defines no {entry}()")
    return function


@dataclass(frozen=True)
class _CompiledSubGoal:
    """Satisfies `hosted_scheduler.SubGoal`: `name`/`judged` as plain attributes, `check` as a bare
    callable. Stored as instance DATA rather than a `def check(self, world, calls)` method so
    `goal.check(world, calls)` invokes the compiled function directly with exactly the two
    positional arguments `_run_phase` passes -- a real method would prepend `self` as a third."""

    name: str
    judged: str
    check: Callable[[Any, Any], object]


@dataclass(frozen=True)
class _CompiledScenario:
    """Satisfies `hosted_scheduler.Scenario`. `scenario_key`/`scenario_id` are carried VERBATIM
    from the document, including an empty `scenario_id` -- synthesizing one here would hide that
    pre-allocation has not actually run (see CONTRACT QUESTIONS: receipts carry `scenario_id ""`
    until that seam is wired). `setup`/`ready` are likewise stored as data, for the same reason as
    `_CompiledSubGoal.check` above."""

    scenario_key: str
    scenario_id: str
    sub_goals: tuple[_CompiledSubGoal, ...]
    setup: Callable[[Any], object]
    ready: Callable[[Any], object]
    # Who this person is and what they came for, as the platform's own persona record. Read off the
    # same document and sent at pre-allocation, so a call can be read on the platform without the
    # scenario file beside it. Presentation only: nothing in the scheduler looks at it.
    presented: dict[str, Any] = field(default_factory=dict)


def _read_text(path: Path, *, label: str) -> str:
    """Missing is "" (mirrors `folder.py`'s own missing-setup/ready-is-empty convention); present
    but unreadable (permission denied, a directory instead of a file) or present but not valid
    UTF-8 is a typed `ScenarioDocumentInvalid`, never a raw `OSError`/`UnicodeDecodeError` escaping
    this module (R1-1) -- both are equally "this scenario folder is malformed", the same
    conclusion `_load_one`'s other reads already reach for a bad `scenario.json`.
    """
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ScenarioDocumentInvalid(f"{label}: cannot read {path.name}: {exc}") from exc


def _validate_subgoal_name(name: str, *, folder_name: str) -> None:
    """R1-3: `sub_goals[]` entries are used verbatim to build `checks/<name>.py` -- a path
    separator or a `..` segment lets a name escape `checks/` (and the sealed bundle) entirely: an
    absolute name execs an arbitrary file never hashed into the bundle's `files[]` (bypassing the
    §2e integrity seal), and a `../`-style traversal that resolves to nothing silently turns into a
    JUDGED sub-goal (`check_path.is_file()` is False) instead of a typed failure. Rejecting
    anything but a plain filename component closes both."""
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise ScenarioDocumentInvalid(
            f"{folder_name}: sub_goals name {name!r} is not a plain filename "
            "(no path separators, no '..', no leading '/')"
        )


def _presented(body: dict[str, Any], *, scenario_key: str) -> dict[str, Any]:
    """The persona record the platform stores for a scenario, built by the platform module's own
    helper rather than a second copy of it here, so both paths describe a person the same way.

    Presentation only, and never load-bearing: anything missing from the document is simply absent
    from what the platform shows, and a document this cannot read still runs.
    """
    from types import SimpleNamespace

    from .platform import persona_of

    try:
        return persona_of(
            SimpleNamespace(
                persona=body.get("persona") or {},
                name=str(body.get("name") or ""),
                scenario_key=scenario_key,
                instruction=str(body.get("instruction") or ""),
                tests=str(body.get("tests") or ""),
            )
        )
    except Exception:  # noqa: BLE001 - a scenario never fails to run over how it is displayed
        return {}


def _deterministic_names(bundle_dir: Path) -> set[str]:
    """Sub-goals the catalogue settles in code, so a missing check file is a defect and not a judge.

    Absence of `checks/<name>.py` is what marks a sub-goal judged, which is right when the catalogue
    says nobody can settle it in code and wrong when the check simply never reached the folder: the
    run then reports a judged verdict for something that was meant to be measured, and a judge asked
    about state it cannot see tends to say yes.
    """
    path = bundle_dir / CATALOGUE
    if not path.is_file():
        return set()
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
        return {
            str(one.get("name") or "")
            for one in (body.get("sub_goals") or [])
            if str(one.get("check") or "").strip()
        } - {""}
    except Exception:  # noqa: BLE001 - an unreadable catalogue leaves the old behaviour
        return set()


def _load_one(folder: Path, *, settled_in_code: set[str] | None = None) -> _CompiledScenario:
    """One scenario folder -> a `Scenario`-protocol object. Mirrors `folder.py`'s documented
    layout (`scenario.json` + `setup.py` + `ready.py` + `checks/<goal>.py`) but reads
    `scenario.json` itself as a plain dict rather than through `fi.alk.harness.scenario.Scenario`
    -- see the module docstring. `folder.py`'s `read_folder` restores only `setup_code`/
    `ready_code` from a folder; it does not read `checks/` at all, so every `checks/<goal>.py` for
    each name in the document's `sub_goals` is read here, by this module, directly.
    """
    body_path = folder / _SCENARIO_JSON
    try:
        raw = body_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        # UnicodeDecodeError is a ValueError, not an OSError -- widened alongside it (R1-1) so a
        # non-UTF-8 `scenario.json` is the same typed failure as an unreadable one, not an escape.
        raise ScenarioDocumentInvalid(
            f"{folder.name}: cannot read {_SCENARIO_JSON}: {exc}"
        ) from exc
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ScenarioDocumentInvalid(
            f"{folder.name}: {_SCENARIO_JSON} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(body, dict):
        raise ScenarioDocumentInvalid(f"{folder.name}: {_SCENARIO_JSON} is not a JSON object")

    scenario_key = body.get("scenario_key", "")
    if not isinstance(scenario_key, str):
        raise ScenarioDocumentInvalid(f"{folder.name}: scenario_key is not a string")
    scenario_id = body.get("scenario_id", "")
    if not isinstance(scenario_id, str):
        raise ScenarioDocumentInvalid(f"{folder.name}: scenario_id is not a string")
    sub_goal_names = body.get("sub_goals", [])
    if not isinstance(sub_goal_names, list) or not all(
        isinstance(name, str) for name in sub_goal_names
    ):
        raise ScenarioDocumentInvalid(f"{folder.name}: sub_goals is not a list of strings")
    for name in sub_goal_names:
        _validate_subgoal_name(name, folder_name=folder.name)

    setup_code = _read_text(folder / _SETUP_PY, label=folder.name)
    ready_code = _read_text(folder / _READY_PY, label=folder.name)
    setup = _compile_entry(setup_code, label=f"{folder.name}/{_SETUP_PY}", entry="setup")
    ready = _compile_entry(ready_code, label=f"{folder.name}/{_READY_PY}", entry="ready")

    sub_goals: list[_CompiledSubGoal] = []
    for name in sub_goal_names:
        check_path = folder / _CHECKS_DIRNAME / f"{name}.py"
        if check_path.is_file():
            check_code = _read_text(check_path, label=folder.name)
            check = _compile_entry(
                check_code, label=f"{folder.name}/{_CHECKS_DIRNAME}/{name}.py", entry="check",
                allow_empty=False,  # R1-2: an existing-but-empty check file is invalid, never a
                                    # vacuous pass -- absence of the file is what means "judged".
            )
            judged = ""
        elif name in (settled_in_code or set()):
            # The catalogue settles this one in code and the file is not here, so the check was
            # written and never materialised. Reporting it as judged is the expensive wrong answer:
            # the scenario runs, the checkpoint reads as assessed, and nothing measured it.
            raise ScenarioDocumentInvalid(
                f"{folder.name}: sub-goal {name!r} is settled in code by the catalogue but "
                f"{_CHECKS_DIRNAME}/{name}.py is missing, so nothing would measure it"
            )
        else:
            # No `checks/<name>.py` -- per `write_folder`'s own `deterministic()` filter, this
            # name is a JUDGED sub-goal.
            judged = _JUDGED_MARKER
            check = _judged_placeholder_check
        sub_goals.append(_CompiledSubGoal(name=name, judged=judged, check=check))

    return _CompiledScenario(
        scenario_key=scenario_key,
        scenario_id=scenario_id,
        sub_goals=tuple(sub_goals),
        setup=setup,
        ready=ready,
        presented=_presented(body, scenario_key=scenario_key),
    )


def load_scenarios(bundle_dir: Path) -> list[_CompiledScenario]:
    """Every scenario document under `<bundle_dir>/scenarios/`, compiled and wrapped, in the same
    sorted-by-folder-name order `folder.py`'s `read_all` uses. Raises `ScenarioDocumentInvalid` on
    the FIRST unreadable or malformed folder -- unlike `read_all`, which skips one and continues;
    a hosted job has nobody watching a suite by hand to notice a scenario silently missing from the
    count, so a folder this reader cannot use fails the whole job instead of shrinking it quietly.
    """
    root = bundle_dir / SCENARIOS_DIRNAME
    if not root.is_dir():
        raise ScenarioDocumentInvalid(f"{root} is not a directory")
    try:
        entries = sorted(root.iterdir())
    except OSError as exc:
        # An unreadable `scenarios/` directory is the same typed failure as any other malformed
        # document (R1-1) -- this is inside `run_job`'s `try`/`except ScenarioDocumentInvalid`
        # (unlike `bundle_has_scenarios`'s own guard above), so raising here is the safe direction.
        raise ScenarioDocumentInvalid(f"{root}: cannot list scenario folders: {exc}") from exc
    settled_in_code = _deterministic_names(bundle_dir)
    scenarios: list[_CompiledScenario] = []
    for folder in entries:
        if not folder.is_dir():
            continue
        scenarios.append(_load_one(folder, settled_in_code=settled_in_code))
    if not scenarios:
        raise ScenarioDocumentInvalid(f"{root} contains no scenario folders")
    return scenarios


class BundleScenarioSource:
    """The real `ScenarioSource`: reads and compiles the bundle's own scenario documents.
    `hosted_entrypoint.run_job` wires this in only when the injected source is still the default
    `NotWiredScenarioSource` AND the bundle actually carries a `scenarios/` directory (the LAYOUT
    DECISION's presence test) -- an injected `ScenarioSource` (every test, every future caller)
    always wins over this one.
    """

    async def build(
        self,
        job: Any,
        bundle: Any,
        scenarios_client: "ScenariosClient",
        *,
        pool: Any,
        world_factory: Any,
        bundle_dir: Path,
    ) -> Sequence[_CompiledScenario]:
        del bundle, pool, world_factory
        # `Path.read_text`/`iterdir`/`compile` are all blocking filesystem+CPU work -- run off the
        # event loop the same way `hosted_entrypoint.py` already does for `bundle_source.load` and
        # `preflight_bundle`, rather than stalling every other in-flight scenario behind it.
        try:
            scenarios = await asyncio.wait_for(
                asyncio.to_thread(load_scenarios, bundle_dir), timeout=_LOAD_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError as exc:
            # R1-5: the underlying thread cannot actually be canceled/killed -- it is left running
            # in the background. Converting the hang into a typed terminal here is still strictly
            # better than the pre-fix behavior (no terminal event, ever): the job gets an honest,
            # bounded FAILED verdict instead of hanging until the platform's own wall clock gives up.
            raise ScenarioDocumentInvalid(
                f"{bundle_dir / SCENARIOS_DIRNAME}: loading scenario documents exceeded "
                f"{_LOAD_TIMEOUT_SECONDS:.0f}s"
            ) from exc
        # An empty `scenario_key` is carried VERBATIM off the document by design (module docstring
        # -- never synthesized here), but it is also the one shape `hosted_entrypoint.py`'s own
        # downstream `_validate_scenarios` would reject as a local, deterministic ENVIRONMENT-domain
        # content defect -- checked HERE, before `register_with_platform` ever reaches the network,
        # so that cheaper, existing local classification wins over a round trip that would only
        # rediscover the same defect as a `platform_sync` failure instead (Azain's serializer
        # rejects a blank `scenario_key` with its own 400 -- `scenario_key` is a plain
        # non-`allow_blank` `CharField`, hosted_harness.py:169). `ScenarioDocumentInvalid` reuses
        # `run_job`'s EXISTING `except ScenarioDocumentInvalid` clause (domain=environment) --
        # nothing new to catch there.
        if any(not scenario.scenario_key for scenario in scenarios):
            raise ScenarioDocumentInvalid(
                f"{bundle_dir / SCENARIOS_DIRNAME}: a scenario document has no non-empty "
                "scenario_key"
            )
        # p13: pre-allocation, after load and before the scheduler ever sees a scenario (spine
        # step 3.5) -- `register_with_platform` raises `ScenarioPreallocationError`/
        # `ob.HostedFencedError`/`ob.HostedChannelFailedError`/`ob.HostedAttemptSupersededError` on
        # any failure, all of which `hosted_entrypoint.run_job`'s existing call site around
        # `scenario_source.build()` already maps to the typed `validating_scenarios`/`platform_sync`
        # terminal (or the fenced exit) -- nothing new to catch here.
        # Pre-allocate the WHOLE suite, then run a sample of it.
        #
        # The sample used to be taken first, so the platform was handed five personas for a suite of
        # thirty and refused the job: "expected exactly 30 personas, got 5". Pre-allocation is sealed
        # against the full set by design (`_begin_payload` sends every key and the platform 409s on a
        # subset), so the suite is what gets registered and the sample is only what gets called. The
        # rows that are not called stay unstarted, which is a truthful state rather than a broken job.
        chosen_evals, agent_prompt, modality = _chosen_evals_and_prompt(bundle_dir)
        registered = await register_with_platform(
            scenarios_client,
            scenarios,
            run_name=job.run_id,
            chosen_evals=chosen_evals,
            agent_prompt=agent_prompt,
            modality=modality,
        )
        return sampled_for_calling(registered)


def _chosen_evals_and_prompt(bundle_dir: Path) -> tuple[list[str], str, str]:
    """The eval names the contract chose, the agent's own prompt, and its modality, for pre-allocation.

    Read as plain JSON rather than through ``AgentContract``: neither value changes how a scenario
    runs, so a contract this cannot parse must cost the run nothing.
    """
    try:
        body = json.loads((bundle_dir / "contract.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a run never fails over what it tells the platform about itself
        return [], "", ""
    if not isinstance(body, dict):
        return [], "", ""
    chosen = body.get("chosen_evals")
    names = (
        [str(one).strip() for one in chosen if str(one).strip()]
        if isinstance(chosen, list)
        else []
    )
    # The modality decides how the platform binds every eval's inputs: a spoken conversation is the
    # recording, a written one is the transcript. Provisioning defaults to text when nobody says, so
    # a voice run that stays quiet here has its evals judge a transcript instead of the call.
    modality = str(body.get("modality") or "").strip().lower()
    return (
        names,
        str(body.get("system_prompt_excerpt") or "").strip(),
        modality if modality in ("voice", "text") else "",
    )


def _preallocation_error(code: str, message: str) -> Exception:
    """Builds a `hosted_entrypoint.ScenarioPreallocationError` for a guard failure below --
    imported lazily (not at module level) because `hosted_entrypoint.py` imports THIS module at
    its own top level (`BundleScenarioSource`/`ScenarioDocumentInvalid`/`bundle_has_scenarios`), so
    a top-level import back would be a circular import. Reusing that exact exception class (rather
    than inventing a new one) is what lets these guard failures land on `run_job`'s ALREADY-WIRED
    `except (ScenarioSourceNotWired, ScenarioPreallocationError)` clause with no changes there.
    """
    from .hosted_entrypoint import ScenarioPreallocationError

    return ScenarioPreallocationError(
        ob.ChannelError(ob.ChannelOutcome.PERMANENT_ITEM, FailureDomain.PLATFORM_SYNC, code, message)
    )


def _provision_payload(
    run_name: str,
    scenarios: Sequence[_CompiledScenario],
    chosen_evals: Sequence[str] = (),
    agent_prompt: str = "",
    modality: str = "",
) -> dict[str, Any]:
    """`HarnessScenarioProvisionSerializer`/`HarnessProvisionPersonaSerializer`
    (futureagi/simulate/serializers/hosted_harness.py): `operation`, `name` and `personas` are
    required; each persona's `name`, `role`, `situation`, `outcome` and `persona` are optional and
    are now supplied from the document (`_presented`). They were omitted while this module read
    `scenario.json` for scheduler-facing fields only, and the cost was a platform that could show a
    call but not who was on it or what they came for.
    """
    payload: dict[str, Any] = {
        "operation": "provision",
        "name": run_name,
        # Everything the serializer accepts, where the document had it: name, role, situation,
        # outcome and the persona itself. `scenario_key` is set last because it is the one field the
        # platform matches on and it must be this scenario's, whatever the persona record says.
        "personas": [
            {**scenario.presented, "scenario_key": scenario.scenario_key}
            for scenario in scenarios
        ],
    }
    # Omitted rather than sent empty, so a platform that does not know these fields is unaffected.
    if chosen_evals:
        payload["chosen_evals"] = list(chosen_evals)
    if agent_prompt:
        payload["agent_prompt"] = agent_prompt
    # Omitted rather than sent empty, so an older platform is unaffected. Sending it matters because
    # provisioning defaults to text and binds every eval to the transcript when nobody says otherwise.
    if modality:
        payload["modality"] = modality
    return payload


def _begin_payload(run_test_id: str, scenarios: Sequence[_CompiledScenario]) -> dict[str, Any]:
    """`HarnessScenarioBeginSerializer` (futureagi/simulate/serializers/hosted_harness.py:193-198):
    `scenario_keys` is `allow_empty=False` and REQUIRED, and `begin_scenarios`
    (services/hosted_harness.py:323-329) 409s (`scenario_key_mismatch`) on anything but an EXACT
    match against the full sealed set -- there is no "subset to run" semantics on the real
    platform (Karthik's contract text describes an optional partial-subset `scenario_ids`; the
    live route does not implement that -- CONTRACT NOTES). The full set is sent every time.
    """
    return {
        "operation": "begin",
        "run_test_id": run_test_id,
        "scenario_keys": [scenario.scenario_key for scenario in scenarios],
    }


def _scenario_ids_by_key(
    submitted: Sequence[_CompiledScenario], raw_scenarios: Any
) -> dict[str, str]:
    """Matches the platform's KEYED provision response
    (`{"scenarios": [{"scenario_key", "scenario_id"}, ...]}`,
    futureagi/simulate/serializers/hosted_harness.py:251-260 +
    services/hosted_harness.py:487-501's `_provision_response`) back onto `submitted` BY
    `scenario_key` -- a dict lookup, never a positional zip. A positional zip (matching Karthik's
    documented `scenario_ids` array shape, not what the platform actually returns) would silently
    mismatch scenario_id -> scenario the instant the response order differs from `submitted`'s
    order, which nothing on the wire guarantees. Every check below raises rather than returning a
    partial mapping -- "never partial assignment" per the brief: the caller only gets a mapping
    once it is proven complete (every submitted key present, exactly once) and exact (no
    unrecognized key).
    """
    if not isinstance(raw_scenarios, list):
        raise _preallocation_error(
            "scenarios_provision_response_invalid", "response 'scenarios' is not a list"
        )
    by_key: dict[str, str] = {}
    for entry in raw_scenarios:
        if not isinstance(entry, dict):
            raise _preallocation_error(
                "scenarios_provision_response_invalid", "a 'scenarios' entry is not an object"
            )
        key = entry.get("scenario_key")
        scenario_id = entry.get("scenario_id")
        if not isinstance(key, str) or not key:
            raise _preallocation_error(
                "scenarios_provision_response_invalid",
                "a 'scenarios' entry has no non-empty scenario_key",
            )
        if key in by_key:
            raise _preallocation_error(
                "scenario_registration_duplicate_key",
                f"scenario_key {key!r} appears more than once in the provision response",
            )
        if not isinstance(scenario_id, str) or not scenario_id:
            raise _preallocation_error(
                "scenarios_provision_response_invalid",
                f"scenario_key {key!r} has no non-empty scenario_id",
            )
        by_key[key] = scenario_id

    submitted_keys = [scenario.scenario_key for scenario in submitted]
    unknown = sorted(set(by_key) - set(submitted_keys))
    if unknown:
        raise _preallocation_error(
            "scenario_registration_unknown_key",
            f"provision response named scenario_key(s) never submitted: {unknown}",
        )
    missing = sorted(set(submitted_keys) - set(by_key))
    if missing:
        raise _preallocation_error(
            "scenario_registration_missing",
            f"provision response is missing scenario_key(s): {missing}",
        )
    return by_key


async def register_with_platform(
    scenarios_client: "ScenariosClient",
    scenarios: Sequence[_CompiledScenario],
    *,
    run_name: str,
    chosen_evals: Sequence[str] = (),
    agent_prompt: str = "",
    modality: str = "",
) -> Sequence[_CompiledScenario]:
    """The scenario pre-allocation SEAM, now wired against the platform's real route (a single
    `POST .../scenarios/`, discriminated by a body-level `operation` field -- see
    `ScenariosClient`'s own docstring for the file:line evidence). `.provision()`/`.begin()` are
    blocking network calls (same `ScenariosClient` the rest of `hosted_entrypoint.py` already
    drives off the event loop via `asyncio.to_thread` -- matched here rather than diverging).

    Sequence: provision (get platform-assigned ids, keyed by `scenario_key`) -> match ids back
    onto `scenarios` with hard guards (`_scenario_ids_by_key`, raises before ANY assignment on any
    mismatch) -> begin (seals execution against the FULL scenario_keys set; a begin failure means
    NO scenario in this batch is returned with an id -- the whole call raises, same as a provision
    failure) -> only then build and return the new scenario list with `scenario_id` filled in.
    """
    provision_result = await asyncio.to_thread(
        scenarios_client.provision,
        _provision_payload(run_name, scenarios, chosen_evals, agent_prompt, modality),
    )
    run_test_id = provision_result.get("run_test_id")
    if not isinstance(run_test_id, str) or not run_test_id:
        raise _preallocation_error(
            "scenarios_provision_response_invalid", "provision response has no run_test_id"
        )
    id_by_key = _scenario_ids_by_key(scenarios, provision_result.get("scenarios"))

    await asyncio.to_thread(scenarios_client.begin, _begin_payload(run_test_id, scenarios))

    return tuple(
        replace(scenario, scenario_id=id_by_key[scenario.scenario_key]) for scenario in scenarios
    )
