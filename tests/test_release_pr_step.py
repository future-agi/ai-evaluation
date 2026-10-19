"""The version-bump step, run for real against throwaway git repositories."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml

if shutil.which("bash") is None or shutil.which("git") is None:
    pytest.skip("needs bash and git", allow_module_level=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "publish-pypi.yml"
STEP_NAME = "Open the version-bump pull request"

BRANCH = "chore/bump-0.2.4"
TITLE = "chore(release): bump agent-learning-kit to 0.2.4"
REFUSAL = "must change one line in pyproject.toml and one in uv.lock"
ORPHAN = "exists without an open pull request"
REAL_GIT = shutil.which("git")

PYPROJECT = '[project]\nname = "agent-learning-kit"\nversion = "0.2.3"\n'
OTHER = 'name = "other-package"\nversion = "1.0.0"\n'
PROJECT = (
    'name = "agent-learning-kit"\nversion = "0.2.3"\nsource = { editable = "." }\n'
)
LOCK = f"version = 1\n\n[[package]]\n{OTHER}\n[[package]]\n{PROJECT}"

# Stand-ins for the two programs the step calls, so no test reaches the network.
UV_STANDIN = """\
import re
import sys
from pathlib import Path

VERSION = re.compile(r'^version = "([^"]+)"$', re.MULTILINE)

args = sys.argv[1:]
if args not in (["version", "--short"], ["version", "--bump", "patch", "--frozen"]):
    sys.exit(2)
pyproject = Path("pyproject.toml")
text = pyproject.read_text(encoding="utf-8")
current = VERSION.search(text).group(1)
if args == ["version", "--short"]:
    print(current)
else:
    major, minor, patch = current.split(".")
    bumped = f"{major}.{minor}.{int(patch) + 1}"
    pyproject.write_text(VERSION.sub(f'version = "{bumped}"', text), encoding="utf-8")
"""

GH_STANDIN = """\
import json
import os
import sys

args = sys.argv[1:]
with open(os.environ["GH_CALLS"], "a", encoding="utf-8") as calls:
    print(json.dumps(args), file=calls)
if args[:2] == ["pr", os.environ.get("FAIL_GH", "")]:
    sys.exit(1)
if args[:2] == ["pr", "list"]:
    print(os.environ.get("EXISTING_PR", ""))
"""

# Passes every call to the real git, except the one subcommand a test asks to fail.
GIT_STANDIN = f"""\
import os
import sys

if sys.argv[1:2] == [os.environ.get("FAIL_GIT", "")]:
    sys.exit(128)
os.execv({REAL_GIT!r}, [{REAL_GIT!r}, *sys.argv[1:]])
"""


class Bench(NamedTuple):
    origin: Path
    clone: Path
    calls: Path
    env: dict[str, str]
    main: str


def _environment(root: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GIT_", "GH_", "GITHUB_"))
    }
    env.update(
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_CONFIG_NOSYSTEM="1",
        HOME=str(root),
        PATH=os.pathsep.join([str(root / "bin"), env.get("PATH", os.defpath)]),
        GITHUB_REPOSITORY="future-agi/agent-learning-kit",
        GH_TOKEN="x",
        GH_CALLS=str(root / "gh-calls.jsonl"),
        EXISTING_PR="",
    )
    return env


def _git(where: Path, env: dict[str, str], *args: str) -> str:
    done = subprocess.run(
        ["git", *args],
        check=False,
        cwd=where,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def _install(path: Path, source: str) -> None:
    path.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
    path.chmod(0o755)


def _make_bench(root: Path, lock: str = LOCK) -> Bench:
    env = _environment(root)
    origin, clone, bin_dir = root / "origin.git", root / "clone", root / "bin"
    bin_dir.mkdir()
    _install(bin_dir / "uv", UV_STANDIN)
    _install(bin_dir / "gh", GH_STANDIN)
    _install(bin_dir / "git", GIT_STANDIN)
    origin.mkdir()
    _git(origin, env, "init", "--bare", "-b", "main")
    clone.mkdir()
    _git(clone, env, "init", "-b", "main")
    _git(clone, env, "remote", "add", "origin", str(origin))
    _git(clone, env, "config", "user.email", "dev@example.com")
    _git(clone, env, "config", "user.name", "Fixture Author")
    (clone / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (clone / "uv.lock").write_text(lock, encoding="utf-8")
    _git(clone, env, "add", "pyproject.toml", "uv.lock")
    _git(clone, env, "commit", "-m", "initial")
    _git(clone, env, "push", "origin", "main")
    main = _git(clone, env, "rev-parse", "HEAD").strip()
    return Bench(
        origin, clone, root / "gh-calls.jsonl", {**env, "GITHUB_SHA": main}, main
    )


@pytest.fixture
def bench(tmp_path: Path) -> Bench:
    return _make_bench(tmp_path)


def _step_script() -> str:
    jobs = yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    steps = [s for s in jobs["release-pr"]["steps"] if s.get("name") == STEP_NAME]
    assert len(steps) == 1
    return steps[0]["run"]


def _run_step(
    bench: Bench, existing_pr: str = "", **failing: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", _step_script()],
        check=False,
        cwd=bench.clone,
        env={**bench.env, "EXISTING_PR": existing_pr, **failing},
        capture_output=True,
        text=True,
        timeout=60,
    )


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return f"exit {result.returncode}\n{result.stdout}\n{result.stderr}"


def _bump_branches(bench: Bench) -> list[str]:
    refs = _git(
        bench.origin,
        bench.env,
        "for-each-ref",
        "--format=%(refname:short)",
        "refs/heads/chore/",
    )
    return refs.split()


def _tip(bench: Bench, ref: str) -> str:
    return _git(bench.origin, bench.env, "rev-parse", ref).strip()


def _gh_calls(bench: Bench) -> list[list[str]]:
    if not bench.calls.exists():
        return []
    lines = bench.calls.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _gh(bench: Bench, *verb: str) -> list[list[str]]:
    return [call for call in _gh_calls(bench) if call[: len(verb)] == list(verb)]


def _flag(call: list[str], name: str) -> str:
    return call[call.index(name) + 1]


def _leave_a_bump_branch_behind(bench: Bench) -> str:
    clone, env = bench.clone, bench.env
    _git(clone, env, "switch", "-c", BRANCH)
    (clone / "pyproject.toml").write_text(
        PYPROJECT.replace("0.2.3", "0.2.4"), encoding="utf-8"
    )
    lock = (clone / "uv.lock").read_text(encoding="utf-8")
    (clone / "uv.lock").write_text(
        lock.replace('version = "0.2.3"', 'version = "0.2.4"'), encoding="utf-8"
    )
    _git(clone, env, "commit", "-am", "bump by hand")
    _git(clone, env, "push", "origin", BRANCH)
    _git(clone, env, "switch", "main")
    _git(clone, env, "branch", "-D", BRANCH)
    return _tip(bench, BRANCH)


def _assert_refused(bench: Bench, result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode != 0, _output(result)
    assert REFUSAL in result.stdout + result.stderr, _output(result)
    assert _bump_branches(bench) == []
    assert _gh(bench, "pr", "create") == []
    assert _git(bench.clone, bench.env, "rev-parse", "HEAD").strip() == bench.main


def test_opens_a_bump_pull_request_with_two_changed_lines(bench):
    result = _run_step(bench)

    assert result.returncode == 0, _output(result)
    assert _bump_branches(bench) == [BRANCH]
    assert _tip(bench, "main") == bench.main
    subject, author, email, committer, parents = _git(
        bench.origin,
        bench.env,
        "log",
        "-1",
        "--format=%s%n%an%n%ae%n%cn%n%P",
        BRANCH,
    ).splitlines()
    assert subject == TITLE
    assert (author, email, committer) == (
        "futureagi-release-bot",
        "release-bot@futureagi.com",
        "futureagi-release-bot",
    )
    assert parents == bench.main
    numstat = _git(bench.origin, bench.env, "diff", "--numstat", "main", BRANCH)
    assert numstat == "1\t1\tpyproject.toml\n1\t1\tuv.lock\n"
    pyproject = _git(bench.origin, bench.env, "show", f"{BRANCH}:pyproject.toml")
    lock = _git(bench.origin, bench.env, "show", f"{BRANCH}:uv.lock")
    assert pyproject == PYPROJECT.replace("0.2.3", "0.2.4")
    assert lock == LOCK.replace("0.2.3", "0.2.4")
    assert OTHER in lock
    (create,) = _gh(bench, "pr", "create")
    assert _flag(create, "--repo") == "future-agi/agent-learning-kit"
    assert _flag(create, "--base") == "main"
    assert _flag(create, "--head") == BRANCH
    assert _flag(create, "--title") == TITLE


def test_does_nothing_when_the_pull_request_is_already_open(bench):
    result = _run_step(bench, existing_pr="12")

    assert result.returncode == 0, _output(result)
    assert _bump_branches(bench) == []
    assert _gh(bench, "pr", "create") == []
    (lookup,) = _gh(bench, "pr", "list")
    assert _flag(lookup, "--head") == BRANCH
    assert _flag(lookup, "--base") == "main"
    assert _flag(lookup, "--state") == "open"
    # A fork's pull request with the same branch name must not count as ours.
    assert _flag(lookup, "--json") == "number,isCrossRepository"
    assert "select(.isCrossRepository | not)" in _flag(lookup, "--jq")


def test_refuses_when_the_branch_exists_without_a_pull_request(bench):
    left = _leave_a_bump_branch_behind(bench)

    result = _run_step(bench)

    assert result.returncode != 0, _output(result)
    assert ORPHAN in result.stdout + result.stderr, _output(result)
    assert _tip(bench, BRANCH) == left
    assert _gh(bench, "pr", "create") == []


def test_ignores_a_branch_that_only_ends_with_the_bump_name(bench):
    _git(bench.clone, bench.env, "push", "origin", f"main:refs/heads/old/{BRANCH}")

    result = _run_step(bench)

    assert result.returncode == 0, _output(result)
    assert _bump_branches(bench) == [BRANCH]
    assert len(_gh(bench, "pr", "create")) == 1


@pytest.mark.parametrize(
    "failing",
    [{"FAIL_GH": "list"}, {"FAIL_GIT": "ls-remote"}, {"FAIL_GIT": "push"}],
    ids=["pull-request lookup", "branch lookup", "push"],
)
def test_stops_when_a_tool_fails(bench, failing):
    result = _run_step(bench, **failing)

    assert result.returncode != 0, _output(result)
    assert _bump_branches(bench) == []
    assert _gh(bench, "pr", "create") == []


def test_a_failed_pull_request_call_is_reported_by_the_next_run(bench):
    first = _run_step(bench, FAIL_GH="create")

    assert first.returncode != 0, _output(first)
    assert _bump_branches(bench) == [BRANCH]
    _git(bench.clone, bench.env, "switch", "main")
    _git(bench.clone, bench.env, "branch", "-D", BRANCH)

    second = _run_step(bench)

    assert second.returncode != 0, _output(second)
    assert ORPHAN in second.stdout + second.stderr, _output(second)


def test_refuses_when_the_lock_is_out_of_step(tmp_path):
    bench = _make_bench(tmp_path, LOCK.replace("0.2.3", "0.2.2"))

    _assert_refused(bench, _run_step(bench))


def test_refuses_when_the_lock_lists_the_project_twice(tmp_path):
    bench = _make_bench(tmp_path, f"{LOCK}\n[[package]]\n{PROJECT}")

    _assert_refused(bench, _run_step(bench))


def test_refuses_when_the_bump_changes_anything_else(bench):
    lock = bench.clone / "uv.lock"
    text = lock.read_text(encoding="utf-8")
    lock.write_text(text.replace("1.0.0", "1.0.1"), encoding="utf-8")

    _assert_refused(bench, _run_step(bench))
