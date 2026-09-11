"""Pull / RL control mode — the AGENT drives a live environment via reset/step.

The push lane has the harness drive the agent; artifact-in scores a submitted
artifact. **Pull** inverts control: the agent is a policy ``obs -> action`` that
steps an environment until done, and the score is the environment's reward. This
is the Gym/OpenEnv shape, run live (not replayed).

Deep-contract + simulated: the environments here are deterministic, in-process,
credential-free simulators (so the lane is fully gate-verifiable). A *live*
external env server (an HTTP step/reset endpoint) is the same contract with a
network transport and is deferred to owner infra — it plugs in as another
``Environment`` without changing the driver or the unified Result.

An environment implements:
  * ``reset(spec) -> (state, obs)``
  * ``step(state, action) -> (state, obs, reward, done, info)``
  * ``optimal_action(obs) -> action``  — a reference policy (proves solvability)
  * ``actions`` — the discrete action set

A policy is a callable ``obs -> action`` or a spec dict: ``{"type": "reference"}``
(the env's optimal policy) or ``{"type": "noop"}`` (always the first action).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol


class Environment(Protocol):
    actions: tuple[str, ...]

    def reset(self, spec: Mapping[str, Any]) -> tuple[dict, dict]: ...
    def step(self, state: dict, action: str) -> tuple[dict, dict, float, bool, dict]: ...
    def optimal_action(self, obs: Mapping[str, Any]) -> str: ...


class ReachTargetEnv:
    """1-D navigation: move toward ``target`` from ``start`` within ``max_steps``.

    obs = {pos, target, remaining}. Reward 1.0 the step the agent lands on target
    (then done); 0.0 otherwise. Deterministic + trivially verifiable; the optimal
    policy is "step toward target".
    """

    actions: tuple[str, ...] = ("left", "right", "stay")

    def reset(self, spec: Mapping[str, Any]) -> tuple[dict, dict]:
        state = {
            "pos": int(spec.get("start", 0)),
            "target": int(spec.get("target", 5)),
            "steps": 0,
            "max_steps": int(spec.get("max_steps", 20)),
        }
        return state, self._obs(state)

    def step(self, state: dict, action: str) -> tuple[dict, dict, float, bool, dict]:
        state = dict(state)
        state["pos"] += {"left": -1, "right": 1, "stay": 0}.get(action, 0)
        state["steps"] += 1
        reached = state["pos"] == state["target"]
        done = reached or state["steps"] >= state["max_steps"]
        reward = 1.0 if reached else 0.0
        return state, self._obs(state), reward, done, {"reached": reached}

    def optimal_action(self, obs: Mapping[str, Any]) -> str:
        if obs["pos"] < obs["target"]:
            return "right"
        if obs["pos"] > obs["target"]:
            return "left"
        return "stay"

    @staticmethod
    def _obs(state: Mapping[str, Any]) -> dict:
        return {
            "pos": state["pos"],
            "target": state["target"],
            "remaining": state["max_steps"] - state["steps"],
        }


class GuessNumberEnv:
    """Binary-search style: guess ``secret`` in [low, high] with higher/lower hints.

    obs = {low, high, last, hint, remaining}. Reward 1.0 on the correct guess.
    Optimal policy = guess the midpoint. Action = the integer guess (as str).
    """

    actions: tuple[str, ...] = ()  # any int in range; reference uses midpoint

    def reset(self, spec: Mapping[str, Any]) -> tuple[dict, dict]:
        low, high = int(spec.get("low", 1)), int(spec.get("high", 100))
        state = {
            "low": low, "high": high, "secret": int(spec.get("secret", (low + high) // 3)),
            "last": None, "hint": "go", "steps": 0,
            "max_steps": int(spec.get("max_steps", 12)),
        }
        return state, self._obs(state)

    def step(self, state: dict, action: str) -> tuple[dict, dict, float, bool, dict]:
        state = dict(state)
        try:
            guess = int(action)
        except (TypeError, ValueError):
            guess = state["low"]
        state["steps"] += 1
        state["last"] = guess
        if guess == state["secret"]:
            state["hint"] = "correct"
            return state, self._obs(state), 1.0, True, {"reached": True}
        if guess < state["secret"]:
            state["low"] = guess + 1
            state["hint"] = "higher"
        else:
            state["high"] = guess - 1
            state["hint"] = "lower"
        done = state["steps"] >= state["max_steps"]
        return state, self._obs(state), 0.0, done, {"reached": False}

    def optimal_action(self, obs: Mapping[str, Any]) -> str:
        return str((int(obs["low"]) + int(obs["high"])) // 2)

    @staticmethod
    def _obs(state: Mapping[str, Any]) -> dict:
        return {
            "low": state["low"], "high": state["high"], "last": state["last"],
            "hint": state["hint"], "remaining": state["max_steps"] - state["steps"],
        }


ENVIRONMENTS: dict[str, Callable[[], Environment]] = {
    "reach_target": lambda: ReachTargetEnv(),
    "guess_number": lambda: GuessNumberEnv(),
}


class PullError(ValueError):
    """Raised for an unknown env kind or malformed pull task."""


def resolve_policy(agent: Any, env: Environment) -> Callable[[Mapping[str, Any]], str]:
    """Resolve a policy from a callable or a spec dict (``reference`` / ``noop``)."""

    if callable(agent):
        return agent
    spec = agent if isinstance(agent, Mapping) else {}
    kind = str(spec.get("type", "reference"))
    if kind == "reference":
        return env.optimal_action
    if kind == "noop":
        first = env.actions[0] if env.actions else "0"
        return lambda _obs: first
    raise PullError(f"unknown pull policy {kind!r}; expected callable / reference / noop")


def run_pull(task: Mapping[str, Any], agent: Any) -> dict[str, Any]:
    """Run one agent-driven episode over a simulated environment.

    Returns ``{"result", "raw"}`` (unified Result). The scalar is the cumulative
    reward; ``pass_fail`` records goal-reached; ``raw`` records the trajectory
    length + terminal info.
    """

    env_spec = task.get("env") or {}
    kind = str(env_spec.get("kind") or "")
    if kind not in ENVIRONMENTS:
        return {
            "result": {"scalar": 0.0, "components": {}, "pass_fail": {},
                       "explanation": f"unknown env kind {kind!r}"},
            "raw": {"control": "pull", "infra_error": True, "env_kind": kind},
        }
    env = ENVIRONMENTS[kind]()
    try:
        policy = resolve_policy(agent, env)
    except PullError as exc:
        return {
            "result": {"scalar": 0.0, "components": {}, "pass_fail": {},
                       "explanation": str(exc)},
            "raw": {"control": "pull", "infra_error": True},
        }

    state, obs = env.reset(env_spec.get("spec") or {})
    total = 0.0
    reached = False
    steps = 0
    hard_cap = int((env_spec.get("spec") or {}).get("max_steps", 50)) + 5
    while steps < hard_cap:
        try:
            action = policy(obs)
        except Exception as exc:  # a misbehaving policy fails the episode, not the lane
            return {
                "result": {"scalar": round(total, 6), "components": {"reward": total},
                           "pass_fail": {"goal_reached": False},
                           "explanation": f"policy raised: {exc}"},
                "raw": {"control": "pull", "env_kind": kind, "steps": steps, "policy_error": True},
            }
        state, obs, reward, done, info = env.step(state, str(action))
        total += float(reward)
        steps += 1
        if info.get("reached"):
            reached = True
        if done:
            break

    return {
        "result": {
            "scalar": round(total, 6),
            "components": {"reward": round(total, 6), "steps": float(steps)},
            "pass_fail": {"goal_reached": reached},
            "explanation": f"{'reached' if reached else 'did not reach'} goal in {steps} steps",
        },
        "raw": {"control": "pull", "env_kind": kind, "steps": steps, "reached": reached},
    }
