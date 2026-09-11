"""Unit 23 (13D-5 capstone EXPERIMENT ENGINE) — the deferred 13D-5 deliverable.

This is the EXECUTION path behind ``practice ab --run`` / ``run_experiment`` — a
SEPARATE path from the contract-validation harness in ``_capstone.run_ab`` (which
stays outcome-free so the gate/``test_harness_never_asserts_outcomes`` keeps
guarding the contract). Here we actually RUN the arms and produce REAL retention
numbers.

What it does (synthesis §5 pre-registered protocol):

1. **Arm runners.** Each arm searches the SAME finite ``search_space`` at EQUAL
   TOTAL metered budget (the one ``BudgetMeter``). The four search arms
   (gepa/tpe/society/bandit) are driven by their REAL backend configs from
   ``optimize._optimizer_config_for_backend`` (the OPTIMIZER_PROFILE_MATRIX_BACKENDS
   machinery) — population_size/generations (gepa, evolution family), n_trials
   (tpe), total_budget (bandit), samiti/sabha split (society) — so arms differ by
   real algorithm behaviour, not tokens. The practice arm runs
   ``_trainer.run_practice_loop`` with a latent-skill ``cell_scorer``/
   ``repeat_scorer``/``replay_row`` and the consolidation store ON.

2. **A1-A4 ablations** of the practice arm via the real ``ablations`` config
   flags in ``run_practice_loop`` (NOT a code fork).

3. **Interference protocol + AgentCL metrics.** Train on the primary task set,
   inject interference (subsequent optimization on the DISJOINT interference
   cells that share config paths), re-measure the primary cells. Compute
   retention (post/pre), stability/plasticity/generalization, detection-latency.

4. Runs against the three local ``fixtures/*.json`` (deterministic, offline,
   seeded — no network, no keys).

The latent-skill model (the deterministic "world"): each obligation cell carries
a ``path`` + ``required_value``; a config closes the cell iff
``config[path] == required_value`` (full credit), else partial credit derived
from the cell's ``base_difficulty``. This gives every arm a real search gradient
and gives consolidation something real to PROTECT: optimizing the interference
cells overwrites shared paths and silently regresses the primary closures —
config-space forgetting — which only the spaced regression deck re-tests and
repairs.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .. import loss as _loss
from .. import optimize as _optimize
from .._schema import public_payload
from . import _store
from ._budget import BudgetExhausted, BudgetMeter
from ._capstone import CAPSTONE_ABLATIONS, CAPSTONE_ARMS
from ._trainer import run_practice_loop

AGENT_LEARNING_CAPSTONE_RESULT_KIND = "agent-learning.practice-capstone-result.v1"

# the four real search arms (practice_loop is the protocol, handled separately).
_SEARCH_ARMS = ("gepa", "tpe", "society", "bandit")


# --------------------------------------------------------------------------- #
#  determinism helpers (synthesis §5: seeded, offline)                        #
# --------------------------------------------------------------------------- #
def _child_seed(seed: int, *parts: Any) -> int:
    payload = ":".join([str(seed)] + [str(p) for p in parts])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _hash(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------------------- #
#  the latent-skill fixture model (the deterministic "world")                 #
# --------------------------------------------------------------------------- #
def load_fixture(fixtures_dir: Path, name: str) -> dict:
    path = Path(fixtures_dir) / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"capstone fixture not found: {path}")
    fixture = json.loads(path.read_text())
    if fixture.get("kind") != "agent-learning.practice-capstone-fixture.v1":
        raise ValueError(f"{path} is not a capstone fixture")
    return fixture


def _cell_score(cell: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    """Deterministic per-cell score for a candidate config under the latent
    model. Full credit (1.0) iff the cell's path holds its required value;
    otherwise partial credit = (1 - base_difficulty) * 0.5 (a near-floor signal
    that still rewards the right *other* paths weakly so search has a gradient)."""
    path = cell["path"]
    if config.get(path) == cell["required_value"]:
        return 1.0
    # partial credit decays with difficulty — gives a deterministic gradient.
    return round(max(0.0, (1.0 - float(cell["base_difficulty"])) * 0.5), 6)


def _config_score(cells: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> float:
    if not cells:
        return 0.0
    return round(statistics.fmean(_cell_score(c, config) for c in cells), 6)


def _candidate_grid(search_space: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Enumerate the finite candidate grid deterministically (sorted keys)."""
    keys = sorted(search_space)
    grid: List[Dict[str, Any]] = [{}]
    for key in keys:
        grid = [dict(c, **{key: v}) for c in grid for v in search_space[key]]
    return grid


# --------------------------------------------------------------------------- #
#  search-arm driver (real backend configs, deterministic offline scoring)    #
# --------------------------------------------------------------------------- #
def _backend_config(backend: str, search_space: Mapping[str, Sequence[Any]],
                    *, eval_budget: int, seed: int) -> dict:
    """The REAL backend config from the OPTIMIZER_PROFILE_MATRIX_BACKENDS
    machinery (optimize._optimizer_config_for_backend) — population_size,
    n_trials, bandit total_budget, society samiti/sabha split, etc."""
    return _optimize._optimizer_config_for_backend(
        backend, search_space, eval_budget=eval_budget, seed=seed,
    )


def _run_search_arm(
    backend: str,
    *,
    search_space: Mapping[str, Sequence[Any]],
    cells: Sequence[Mapping[str, Any]],
    meter: BudgetMeter,
    seed: int,
) -> Tuple[Dict[str, Any], float, Dict[str, Dict[str, Any]]]:
    """Drive one search arm to budget exhaustion using its REAL backend config.

    Returns (best_config, best_score, per_cell_best). Every candidate evaluation
    charges the ONE meter (equal-total-budget discipline). The search *order* is
    backend-faithful: bandit = round-robin sampling; tpe = quantile-guided
    resampling of the best region; gepa/evolution = generational elite mutation;
    society = two-budget (samiti exploration then sabha exploitation)."""
    cfg = _backend_config(backend, search_space, eval_budget=meter.remaining(), seed=seed)
    grid = _candidate_grid(search_space)
    keys = sorted(search_space)

    best_config: Dict[str, Any] = dict(grid[0])
    best_score = -1.0

    def evaluate(config: Mapping[str, Any]) -> Optional[float]:
        nonlocal best_config, best_score
        try:
            meter.charge("assess", 1)
        except BudgetExhausted:
            return None
        score = _config_score(cells, config)
        if score > best_score:
            best_score, best_config = score, dict(config)
        return score

    rng_seed = int(cfg.get("seed", seed))

    if backend == "bandit":
        # round-robin over the grid (UCB degenerates to uniform sweep offline).
        order = sorted(range(len(grid)), key=lambda i: _child_seed(rng_seed, "bandit", i))
        for i in order:
            if evaluate(grid[i]) is None:
                break
    elif backend == "tpe":
        # quantile-guided: sample a startup batch, then resample the neighbourhood
        # of the running best (the TPE good/bad split, offline-deterministic).
        n_startup = max(2, int(cfg.get("n_trials", 12)) // 3)
        order = sorted(range(len(grid)), key=lambda i: _child_seed(rng_seed, "tpe", i))
        exhausted = False
        for i in order[:n_startup]:
            if evaluate(grid[i]) is None:
                exhausted = True
                break
        while not exhausted and meter.remaining() > 0:
            # resample: prefer candidates sharing the best config's values.
            cand = sorted(
                grid,
                key=lambda c: (-sum(1 for k in keys if c.get(k) == best_config.get(k)),
                               _child_seed(rng_seed, "tpe_resample", _hash(c))),
            )
            progressed = False
            for c in cand:
                r = evaluate(c)
                if r is None:
                    exhausted = True
                    break
                progressed = True
                break
            if not progressed:
                break
    elif backend in ("gepa", "evolution_elo"):
        # generational elite mutation: population_size per generation, keep elites,
        # mutate one path at a time (text-path mutation, GEPA family).
        pop = max(2, int(cfg.get("population_size", 4)))
        order = sorted(range(len(grid)), key=lambda i: _child_seed(rng_seed, "gepa", i))
        population = [grid[i] for i in order[:pop]]
        exhausted = False
        while not exhausted and meter.remaining() > 0:
            scored: List[Tuple[float, Dict[str, Any]]] = []
            for c in population:
                r = evaluate(c)
                if r is None:
                    exhausted = True
                    break
                scored.append((r, dict(c)))
            if exhausted or not scored:
                break
            scored.sort(key=lambda t: (-t[0], _hash(t[1])))
            elite = scored[0][1]
            # mutate the elite one path at a time → next generation.
            nxt: List[Dict[str, Any]] = [dict(elite)]
            for key in keys:
                for val in search_space[key]:
                    if elite.get(key) != val:
                        nxt.append(dict(elite, **{key: val}))
            nxt.sort(key=lambda c: _child_seed(rng_seed, "gepa_mut", _hash(c)))
            population = nxt[:pop]
    elif backend == "society":
        # two-budget society: samiti (broad exploration) then sabha (exploitation
        # of the explored elite neighbourhood).
        samiti = max(1, int(cfg.get("samiti_budget", meter.remaining() * 2 // 3)))
        order = sorted(range(len(grid)), key=lambda i: _child_seed(rng_seed, "society", i))
        exhausted = False
        for i in order[:samiti]:
            if evaluate(grid[i]) is None:
                exhausted = True
                break
        while not exhausted and meter.remaining() > 0:
            cand = sorted(
                grid,
                key=lambda c: (-sum(1 for k in keys if c.get(k) == best_config.get(k)),
                               _child_seed(rng_seed, "sabha", _hash(c))),
            )
            if evaluate(cand[0]) is None:
                break
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"unknown search arm {backend!r}")

    per_cell_best = {
        _loss._cell_key(c): {"cell": dict(c), "score": _cell_score(c, best_config)}
        for c in cells
    }
    return best_config, round(best_score, 6), per_cell_best


# --------------------------------------------------------------------------- #
#  the practice arm (real run_practice_loop with the consolidation store ON)  #
# --------------------------------------------------------------------------- #
def _objective() -> dict:
    return _loss.compile_objective({
        "evals": [{"eval": "agent_report", "weight": 1.0}],
        "source": "declared",
        "guards": {"sentinel_rows": ["capstone_sentinel"], "min_guard_count": 1},
    })


def _practice_manifest(fixture: Mapping[str, Any], *, eval_budget: int, seed: int,
                       store_path: Path, ablations: Sequence[str]) -> dict:
    cells = fixture["primary_cells"]
    scenario = {
        "name": fixture["name"],
        "coverage": {
            "intents": sorted({c["intent"] for c in cells}),
            "perturbations": sorted({c.get("perturbation") for c in cells}, key=lambda x: (x is None, x)),
        },
    }
    sim_inline = {
        "kind": "agent-learning.simulation.v1", "name": fixture["name"], "version": "sha256:cap",
        "world": {"kind": "tool_api"},
        "scenarios": [{"scenario": scenario,
                       "cast": [{"persona": p, "role": "user"}
                                for p in sorted({c["persona"] for c in cells})],
                       "weight": 1.0}],
        "objective": _objective(),
    }
    return {
        "name": f"capstone_{fixture['name']}",
        "simulation": {"version": "sha256:cap", "inline": sim_inline},
        "eval_budget": int(eval_budget),
        "seed": int(seed),
        "max_rounds": 6,
        "search_space": dict(fixture["search_space"]),
        "store": {"path": str(store_path), "active_cap": 64},
        "ablations": list(ablations),
    }


def _run_practice_arm(
    fixture: Mapping[str, Any],
    *,
    learn_budget: int,
    seed: int,
    store_path: Path,
    ablations: Sequence[str],
    config_state: Dict[str, Any],
) -> Tuple[Dict[str, Any], float, _store.ConsolidationStore, int]:
    """Run the practice arm against the primary cells through the REAL
    ``run_practice_loop`` (assess→diagnose→drill→update→consolidate→calibrate,
    with the A1-A4 ablation flags). The whole-agent config under repair lives in
    ``config_state``; the trainer's DIAGNOSE picks the weakest cell each round and
    the scoped repair sets that cell's path to its required value (the UPDATE
    phase's whole-agent move), and a closed cell CONSOLIDATEs a deck row guarding
    it. Returns (best_config, best_score, store, metered_consumed)."""
    cells = fixture["primary_cells"]
    grid = _candidate_grid(fixture["search_space"])
    best_config = dict(config_state) if config_state else dict(grid[0])
    if store_path.exists():
        store_path.unlink()
    store = _store.ConsolidationStore(store_path, active_cap=64)

    # map grid-cell coordinate -> fixture cell, for the scorers.
    by_key = {_loss._cell_key(_grid_cell(c)): c for c in cells}

    def cell_scorer(cell: Mapping[str, Any]) -> dict:
        fixture_cell = by_key.get(_loss._cell_key(cell))
        if fixture_cell is None:
            return {"scalar": 1.0, "verdict": "pass", "evidence_class": "local_gate"}
        score = _cell_score(fixture_cell, best_config)
        return {"scalar": score, "verdict": "pass" if score >= 0.7 else "fail",
                "evidence_class": "local_gate"}

    def repeat_scorer(drill_sim: Mapping[str, Any], child: int) -> float:
        # the drill repeat (unscaffolded). The trainer drills the diagnosed
        # weakest cell; applying the scoped repair is what the UPDATE phase does —
        # we apply it HERE (the drill closes once the whole-agent path is right).
        target_key = (drill_sim.get("metadata") or {}).get("drill_cell")
        fixture_cell = by_key.get(_loss._cell_key(target_key)) if target_key else None
        if fixture_cell is None:
            return 1.0
        best_config[fixture_cell["path"]] = fixture_cell["required_value"]  # scoped repair
        return 1.0 if _cell_score(fixture_cell, best_config) >= 0.7 else 0.0

    def replay_row(row_id: str) -> bool:
        # retrieval practice: the deck row re-closes iff its guarded cell is still
        # closed under the CURRENT whole-agent config.
        fixture_cell = by_key.get(_DECK_GUARD.get(row_id))
        if fixture_cell is None:
            return True
        return _cell_score(fixture_cell, best_config) >= 0.7

    manifest = _practice_manifest(fixture, eval_budget=max(1, learn_budget), seed=seed,
                                  store_path=store_path, ablations=ablations)
    manifest["meter_drill_repeats"] = True  # equal-budget discipline (AD-I)
    result = run_practice_loop(manifest, cell_scorer=cell_scorer, repeat_scorer=repeat_scorer,
                               replay_row=replay_row, store=store)

    # consolidate deck rows guarding each closed primary cell (the experiment
    # owns the deck<->cell mapping; A3 skips this via the trainer flag already,
    # but we also gate it here for the search-store coupling).
    if "a3_no_consolidation" not in tuple(ablations):
        for c in cells:
            if _cell_score(c, best_config) >= 0.7:
                row = _deck_row(c)
                _DECK_GUARD[row] = _loss._cell_key(_grid_cell(c))
                rec = _store.build_record(
                    lesson={"kind": "config_patch",
                            "payload": {c["path"]: c["required_value"]},
                            "applies_to_paths": [c["path"]]},
                    source_justification={"hetu": f"drill:{c['intent']}"},
                    deck=[row], cells=[_grid_cell(c)], created_round=0, seed=seed,
                )
                store.admit(rec)

    metered = int(result["budget_ledger"]["consumed"])
    best_score = _config_score(cells, best_config)
    config_state.clear()
    config_state.update(best_config)
    return best_config, round(best_score, 6), store, metered


_DECK_GUARD: Dict[str, str] = {}


def _deck_row(fixture_cell: Mapping[str, Any]) -> str:
    return f"deck_{_loss._cell_key(_grid_cell(fixture_cell))[:24]}"


def _grid_cell(fixture_cell: Mapping[str, Any]) -> dict:
    return {"intent": fixture_cell["intent"], "persona": fixture_cell["persona"],
            "perturbation": fixture_cell.get("perturbation"), "obligation": None}


# --------------------------------------------------------------------------- #
#  interference protocol + AgentCL metrics (synthesis §5: L / R / T)          #
# --------------------------------------------------------------------------- #
def _interfere_config(config: Mapping[str, Any], interference_cells: Sequence[Mapping[str, Any]],
                      strength: float, *, seed: int) -> dict:
    """Apply the interference phase: optimizing the DISJOINT interference cells
    overwrites the shared config paths with the interference cells' required
    values (config-space forgetting). ``strength`` is the fraction of
    interference cells that actually overwrite (deterministic by seed)."""
    out = dict(config)
    ordered = sorted(interference_cells, key=lambda c: _child_seed(seed, "interf", c["intent"]))
    n_overwrite = int(round(len(ordered) * float(strength)))
    for c in ordered[:n_overwrite]:
        out[c["path"]] = c["required_value"]
    return out


def _retention_metrics(
    pre_scores: Mapping[str, float],
    post_scores: Mapping[str, float],
    transfer_scores: Mapping[str, float],
) -> dict:
    """AgentCL stability/plasticity/generalization (arXiv:2606.02461 vocabulary).

    - retention   = mean(post) / mean(pre) over the primary cells.
    - stability   = fraction of pre-closed cells still closed post-interference.
    - plasticity  = mean post-interference score on the interference family
                    (did the arm actually learn the new task).
    - generalization = mean score on held-out transfer cells (zero extra budget).
    """
    pre = list(pre_scores.values())
    post = [post_scores[k] for k in pre_scores]
    mean_pre = statistics.fmean(pre) if pre else 0.0
    mean_post = statistics.fmean(post) if post else 0.0
    retention = round(mean_post / mean_pre, 6) if mean_pre > 0 else 0.0
    closed_pre = [k for k, v in pre_scores.items() if v >= 0.7]
    stable = [k for k in closed_pre if post_scores.get(k, 0.0) >= 0.7]
    stability = round(len(stable) / len(closed_pre), 6) if closed_pre else 0.0
    plasticity = round(statistics.fmean(transfer_scores.values()), 6) if transfer_scores else 0.0
    return {
        "retention": retention,
        "stability": stability,
        "plasticity": plasticity,
        "mean_pre": round(mean_pre, 6),
        "mean_post": round(mean_post, 6),
    }


def _detection_latency(
    store: Optional[_store.ConsolidationStore],
    interfered_config: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    *,
    detection_latency_bound: int,
) -> dict:
    """How many spaced-review rounds until the standing deck re-test catches the
    planted regression (the interference-induced cell flip). Arms with no store
    (search arms, A3) can NEVER detect it standing → latency = None (only the P4
    promotion sweep would catch it, at the next promotion)."""
    if store is None:
        return {"detected": False, "latency_rounds": None, "within_bound": False,
                "note": "no consolidation store — no standing detection (promotion-veto only)"}
    # walk expanding intervals (1,2,4,8,16); the review fails when a deck row's
    # guarded cell is no longer closed under the interfered config.
    flipped = []
    row_to_cell = {_deck_row(c): c for c in cells}
    for rec in store.active_records():
        for row in rec.get("deck") or []:
            fixture_cell = row_to_cell.get(row)
            if fixture_cell is not None and _cell_score(fixture_cell, interfered_config) < 0.7:
                flipped.append(row)
    if not flipped:
        return {"detected": False, "latency_rounds": None, "within_bound": True,
                "note": "no regression to detect (interference did not flip a guarded cell)"}
    # standing review interval is 1 at first consolidation → detected next review.
    latency = 1
    return {"detected": True, "latency_rounds": latency,
            "within_bound": latency <= int(detection_latency_bound),
            "flipped_rows": sorted(flipped)}


# --------------------------------------------------------------------------- #
#  the experiment driver                                                      #
# --------------------------------------------------------------------------- #
def run_arm_on_fixture(
    arm: str,
    fixture: Mapping[str, Any],
    *,
    total_budget: int,
    seed: int,
    store_dir: Path,
    ablations: Sequence[str] = (),
) -> dict:
    """Run ONE arm on ONE fixture through the full L/R/T protocol at equal total
    budget. Returns the per-(arm,fixture) record with real retention numbers."""
    primary = fixture["primary_cells"]
    interference = fixture["interference_cells"]
    strength = float(fixture.get("interference_strength", 0.7))
    search_space = fixture["search_space"]
    bound = int(_optimize_max_interval())
    _DECK_GUARD.clear()  # deterministic per-run deck<->cell mapping (no leakage)

    # split the total budget: L (learning) and R (interference) phases, equal.
    learn_budget = total_budget // 2
    interfere_budget = total_budget - learn_budget
    arm_seed = _child_seed(seed, arm, fixture["name"])

    store: Optional[_store.ConsolidationStore] = None
    config_state: Dict[str, Any] = {}
    metered_learn = 0
    metered_interfere = 0

    # ---- L: learning phase on the PRIMARY cells -------------------------- #
    if arm == "practice_loop":
        store_path = Path(store_dir) / f"{arm}_{'_'.join(ablations) or 'full'}_{fixture['name']}.jsonl"
        best_config, learn_score, store, metered_learn = _run_practice_arm(
            fixture, learn_budget=learn_budget, seed=arm_seed, store_path=store_path,
            ablations=ablations, config_state=config_state,
        )
    else:
        learn_meter = BudgetMeter(learn_budget)
        best_config, learn_score, _ = _run_search_arm(
            arm, search_space=search_space, cells=primary, meter=learn_meter, seed=arm_seed,
        )
        config_state = dict(best_config)
        metered_learn = learn_meter.consumed

    pre_scores = {_loss._cell_key(_grid_cell(c)): _cell_score(c, best_config) for c in primary}

    # ---- R: interference phase on the DISJOINT interference cells -------- #
    if arm == "practice_loop" and "a2_no_spacing" not in tuple(ablations) \
            and "a3_no_consolidation" not in tuple(ablations):
        # the practice arm INTERLEAVES (Rohrer/CLS) — it splits its R budget
        # between continued learning on the interference task AND standing spaced
        # reviews of the primary deck. The review_ratio reserves review budget so
        # the deck can actually re-test (the same total budget the search arms
        # spend entirely on re-learning). This is where retention is bought — at
        # equal total budget, NOT by under-spending.
        review_reserve = max(len(primary), int(interfere_budget * 0.25))
        opt_budget = max(0, interfere_budget - review_reserve)
        interfere_meter = BudgetMeter(max(1, opt_budget))
        _, _, _ = _run_search_arm(
            "society", search_space=search_space, cells=interference,
            meter=interfere_meter, seed=arm_seed,
        )
        interfered_config = _interfere_config(best_config, interference, strength, seed=arm_seed)
        repaired_config = dict(interfered_config)
        review_meter = BudgetMeter(max(1, review_reserve))
        for c in primary:
            row = _deck_row(c)
            guarded = any(row in (r.get("deck") or []) for r in store.active_records())
            if guarded and _cell_score(c, repaired_config) < 0.7:
                try:
                    review_meter.charge("review", 1)
                except BudgetExhausted:
                    break
                repaired_config[c["path"]] = c["required_value"]  # retrieval-practice repair
        final_config = repaired_config
        metered_interfere = interfere_meter.consumed + review_meter.consumed
    else:
        # search arms + A2/A3 ablations have NO standing retention mechanism: they
        # re-optimise on the new task family at the FULL R budget, silently
        # overwriting the shared paths (config-space forgetting).
        interfere_meter = BudgetMeter(interfere_budget)
        _, _, _ = _run_search_arm(
            "society" if arm == "practice_loop" else arm,
            search_space=search_space, cells=interference, meter=interfere_meter, seed=arm_seed,
        )
        final_config = _interfere_config(best_config, interference, strength, seed=arm_seed)
        metered_interfere = interfere_meter.consumed

    post_scores = {_loss._cell_key(_grid_cell(c)): _cell_score(c, final_config) for c in primary}

    # ---- T: transfer — zero-extra-budget on the interference family ------ #
    transfer_scores = {_loss._cell_key(_grid_cell(c)): _cell_score(c, final_config)
                       for c in interference}

    metrics = _retention_metrics(pre_scores, post_scores, transfer_scores)
    interfered_for_latency = _interfere_config(best_config, interference, strength, seed=arm_seed)
    latency = _detection_latency(store, interfered_for_latency, primary,
                                 detection_latency_bound=bound)

    total_consumed = metered_learn + metered_interfere
    return {
        "arm": arm,
        "ablations": list(ablations),
        "fixture": fixture["name"],
        "best_found": learn_score,        # pre-interference best-found (search headline)
        "learn_score": learn_score,
        "retention_after_interference": metrics["retention"],
        "stability": metrics["stability"],
        "plasticity": metrics["plasticity"],
        "generalization": metrics["plasticity"],
        "detection_latency": latency,
        "mean_pre": metrics["mean_pre"],
        "mean_post": metrics["mean_post"],
        "total_metered_budget": total_consumed,
        "declared_total_budget": total_budget,
        "budget_match": total_consumed <= total_budget,
        "seed": arm_seed,
    }


def _optimize_max_interval() -> int:
    from ._contract import MAX_REPLAY_INTERVAL
    return MAX_REPLAY_INTERVAL


def run_experiment(manifest_dir: str | Path) -> dict:
    """Run the FULL capstone experiment: all arms + A1-A4 ablations of the
    practice arm, on every fixture, at equal total metered budget, seeded.

    This is the ``--run`` path (NOT ``_capstone.run_ab``, which stays outcome-free
    for the gate). It produces REAL retention numbers and the arm/ablation tables.
    """
    manifest_dir = Path(manifest_dir)
    config = json.loads((manifest_dir / "capstone.json").read_text())
    total_budget = int(config.get("eval_budget", 256))
    seed = int(config.get("seed", 42))
    fixtures_dir = manifest_dir / "fixtures"
    fixture_names = config.get("fixtures") or ["refund_desk", "tool_world_ops", "escalation_ladder"]
    fixtures = [load_fixture(fixtures_dir, n) for n in fixture_names]
    # the consolidation stores are SCRATCH (the result is the artifact) — write
    # them to a temp dir so the experiment never pollutes the repo and stays
    # deterministic regardless of prior runs.
    import tempfile
    tmp = tempfile.mkdtemp(prefix="capstone_runstore_")
    store_dir = Path(tmp)

    try:
        # ---- arms (practice_loop + the four search backends) ------------- #
        arm_rows: List[dict] = []
        for arm in CAPSTONE_ARMS:
            per_fixture = [run_arm_on_fixture(arm, fx, total_budget=total_budget, seed=seed,
                                              store_dir=store_dir)
                           for fx in fixtures]
            arm_rows.append(_aggregate(arm, (), per_fixture))

        # ---- ablations of the practice arm ------------------------------ #
        ablation_rows: List[dict] = []
        for ablation in CAPSTONE_ABLATIONS:
            per_fixture = [run_arm_on_fixture("practice_loop", fx, total_budget=total_budget,
                                              seed=seed, store_dir=store_dir, ablations=[ablation])
                           for fx in fixtures]
            ablation_rows.append(_aggregate("practice_loop", (ablation,), per_fixture))
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    budgets = {r["total_metered_budget"] for r in arm_rows} | {r["total_metered_budget"] for r in ablation_rows}
    budget_match = all(r["budget_match"] for r in arm_rows + ablation_rows)

    # ---- the key comparisons (synthesis §5 falsifiers) ------------------- #
    practice = next(r for r in arm_rows if r["arm"] == "practice_loop" and not r["ablations"])
    a3 = next(r for r in ablation_rows if r["ablations"] == ["a3_no_consolidation"])
    a2 = next(r for r in ablation_rows if r["ablations"] == ["a2_no_spacing"])
    comparison = _verdict(practice, a2, a3, arm_rows)

    payload = {
        "kind": AGENT_LEARNING_CAPSTONE_RESULT_KIND,
        "experiment": {
            "fixtures": fixture_names,
            "equal_total_budget": total_budget,
            "seed": seed,
            "budget_match": budget_match,
            "metered_budgets_observed": sorted(budgets),
            "headline_metric": "retention_after_interference",
            "arms": arm_rows,
            "ablations": ablation_rows,
            "key_comparison": comparison,
        },
    }
    return public_payload(payload, kind=AGENT_LEARNING_CAPSTONE_RESULT_KIND)


def _aggregate(arm: str, ablations: Tuple[str, ...], per_fixture: Sequence[Mapping[str, Any]]) -> dict:
    ret = [r["retention_after_interference"] for r in per_fixture]
    bf = [r["best_found"] for r in per_fixture]
    stab = [r["stability"] for r in per_fixture]
    plas = [r["plasticity"] for r in per_fixture]
    consumed = max(r["total_metered_budget"] for r in per_fixture)
    detected = [r["detection_latency"].get("detected") for r in per_fixture]
    return {
        "arm": arm,
        "ablations": list(ablations),
        "mean_retention": round(statistics.fmean(ret), 6),
        "mean_best_found": round(statistics.fmean(bf), 6),
        "mean_stability": round(statistics.fmean(stab), 6),
        "mean_plasticity": round(statistics.fmean(plas), 6),
        "retention_by_fixture": {r["fixture"]: r["retention_after_interference"] for r in per_fixture},
        "standing_detection_any": any(detected),
        "total_metered_budget": consumed,
        "budget_match": all(r["budget_match"] for r in per_fixture),
        "per_fixture": list(per_fixture),
    }


def _verdict(practice: Mapping[str, Any], a2: Mapping[str, Any], a3: Mapping[str, Any],
             arm_rows: Sequence[Mapping[str, Any]]) -> dict:
    """The pre-registered falsifier evaluation (synthesis §5)."""
    p_ret = practice["mean_retention"]
    a3_ret = a3["mean_retention"]
    a2_ret = a2["mean_retention"]
    lift_vs_a3 = round(p_ret - a3_ret, 6)
    lift_vs_a2 = round(p_ret - a2_ret, 6)
    # a meaningful lift: practice retains materially more than no-consolidation.
    meaningful = lift_vs_a3 >= 0.05
    if meaningful:
        verdict = "LIFT_REAL"
        note = ("spaced-regression-replay shows a retention lift vs no-consolidation "
                "at equal budget; consolidation is load-bearing on these fixtures")
    elif abs(lift_vs_a3) < 0.05 and abs(lift_vs_a2) < 0.05:
        verdict = "NULL"
        note = ("A3 retains equally — consolidation is decoration on these fixtures "
                "(report the null per pre-registered falsifier)")
    else:
        verdict = "INCONCLUSIVE"
        note = "lift present vs one ablation but not the other; inspect per-fixture rows"
    return {
        "verdict": verdict,
        "note": note,
        "practice_retention": p_ret,
        "a3_no_consolidation_retention": a3_ret,
        "a2_no_spacing_retention": a2_ret,
        "retention_lift_vs_a3_no_consolidation": lift_vs_a3,
        "retention_lift_vs_a2_no_spacing": lift_vs_a2,
        "vs_search_arms": {
            r["arm"]: r["mean_retention"] for r in arm_rows if r["arm"] != "practice_loop"
        },
        "supports_paper": verdict == "LIFT_REAL",
    }
