"""Deterministic container packaging for repositories that ship no container metadata.

The adapter packages a submitted process; it never generates application behavior.  It accepts
only dependency and entrypoint conventions that can be proven from files in the repository, and
stages a sanitized build context outside the submitted checkout so generated files and secret
configuration never mutate or enter the customer's source tree.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from packaging.utils import canonicalize_name

from .public_material import is_public_pem, inspect_public_file, is_public_npm_config

GENERATED_DOCKERFILE = ".alk-generated.Dockerfile"
GENERATED_PLAN = "generated-runtime.json"
_MAX_FILES = 20_000
_MAX_FILE_BYTES = 64 * 1024 * 1024
_MAX_CONTEXT_BYTES = 512 * 1024 * 1024
_IGNORED_DIRECTORIES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
}
_SECRET_NAMES = {
    ".env",
    ".npmrc",
    ".pypirc",
    "credentials",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
}
_SECRET_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}

_PYTHON_ENTRYPOINTS = ("agent.py", "main.py", "app.py", "bot.py", "server.py")
_NODE_ENTRYPOINTS = ("agent.js", "index.js", "main.js", "server.js")


class GeneratedRuntimeError(ValueError):
    """The repository cannot be packaged without unsafe or ambiguous guessing."""


@dataclass(frozen=True)
class PythonExtraSelection:
    names: tuple[str, ...] = ()
    all_packages: bool = False

    @property
    def uv_arguments(self) -> list[str]:
        return (["--all-packages"] if self.all_packages else []) + [
            value for name in self.names for value in ("--extra", name)
        ]


@dataclass(frozen=True)
class GeneratedRuntimePlan:
    language: str
    component: str
    version: str
    dependency_file: str
    install_strategy: str
    extras: tuple[str, ...]
    command: tuple[str, ...]
    context_excludes: tuple[str, ...]
    dockerfile: str
    context_directory: str = ""

    @property
    def fingerprint(self) -> str:
        payload = asdict(self)
        payload.pop("context_directory", None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def can_generate_runtime(source: str | Path, runtime: Any | None = None) -> bool:
    """Whether a deterministic language manifest exists; this performs no writes."""
    try:
        detect_generated_runtime(source, runtime)
    except GeneratedRuntimeError:
        return False
    return True


def detect_generated_runtime(
    source: str | Path, runtime: Any | None = None
) -> GeneratedRuntimePlan:
    """Return an auditable runtime plan, or one actionable admission error."""
    if (
        getattr(runtime, "execution", "auto") == "compose"
        or getattr(runtime, "compose_file", "")
        or getattr(runtime, "service", "")
        or getattr(runtime, "required_environment", {})
    ):
        raise GeneratedRuntimeError(
            "runtime_selection_unsupported: generated runtime cannot enforce selected Compose services or process obligations"
        )
    root = Path(source).expanduser().resolve()
    component = _component_root(root, str(getattr(runtime, "workdir", "") or ""))
    language = str(getattr(runtime, "language", "") or "").strip().lower()
    python_markers = _python_markers(component)
    node_marker = component / "package.json"

    if language in {"python", "py"}:
        if not python_markers:
            raise GeneratedRuntimeError(
                "generated_runtime_dependency_manifest_missing: expected pyproject.toml, "
                "requirements.txt, or a Python lock file"
            )
        language = "python"
    elif language in {"node", "nodejs", "javascript", "typescript", "js", "ts"}:
        if not node_marker.is_file():
            raise GeneratedRuntimeError(
                "generated_runtime_dependency_manifest_missing: expected package.json"
            )
        language = "node"
    elif python_markers and not node_marker.is_file():
        language = "python"
    elif node_marker.is_file() and not python_markers:
        language = "node"
    elif python_markers and node_marker.is_file():
        raise GeneratedRuntimeError(
            "generated_runtime_language_ambiguous: both Python and Node manifests exist; "
            "set runtime.language and runtime.workdir to the intended component"
        )
    else:
        nested = _nested_components(component)
        if len(nested) == 1 and component == root:
            component = nested[0]
            return detect_generated_runtime(
                root,
                _runtime_override(
                    runtime, workdir=component.relative_to(root).as_posix()
                ),
            )
        if len(nested) > 1:
            choices = ", ".join(
                path.relative_to(root).as_posix() for path in nested[:12]
            )
            raise GeneratedRuntimeError(
                "generated_runtime_component_ambiguous: multiple runnable components found: "
                f"{choices}; set runtime.workdir"
            )
        raise GeneratedRuntimeError(
            "generated_runtime_dependency_manifest_missing: no supported Python or Node "
            "dependency manifest was found"
        )

    explicit_command = _explicit_command(runtime)
    relative_component = component.relative_to(root).as_posix()
    if relative_component == ".":
        relative_component = ""
    if language == "python":
        version = _python_language_version(
            component, str(getattr(runtime, "version", "") or "")
        )
        command = explicit_command or _python_command(component)
        dependency, strategy, install, extras = _python_install(
            component, runtime, command
        )
        requirements = select_python_requirements(
            component,
            list(getattr(runtime, "requirements_files", None) or []),
            context_excludes=_context_excludes(component, runtime, dependency, command),
        )
        if requirements:
            interpreter = (
                "/app/.venv/bin/python" if strategy.startswith("uv-") else "python"
            )
            install.append("RUN pip install --no-cache-dir uv==0.8.13")
            install.append(
                "RUN "
                + shlex.join(
                    supplemental_requirements_command(interpreter, requirements)
                )
            )
            install.append(
                "RUN "
                + shlex.join(
                    [
                        "uv",
                        "pip",
                        "check",
                        "--python",
                        interpreter,
                    ]
                )
            )
        dockerfile = _python_dockerfile(version, dependency, strategy, install, command)
    else:
        version = _language_version(str(getattr(runtime, "version", "") or ""), "22")
        dependency, strategy, install, runner = _node_install(component)
        command = explicit_command or _node_command(component, runner)
        install.extend(_node_build_steps(component, runner, command))
        dockerfile = _node_dockerfile(version, dependency, install, command)
        extras = []
    return GeneratedRuntimePlan(
        language=language,
        component=relative_component,
        version=version,
        dependency_file=dependency,
        install_strategy=strategy,
        extras=tuple(extras),
        command=tuple(command),
        context_excludes=_context_excludes(component, runtime, dependency, command),
        dockerfile=dockerfile,
    )


def prepare_generated_runtime(
    source: str | Path, destination: str | Path, runtime: Any | None = None
) -> GeneratedRuntimePlan:
    """Detect the runtime and create its sanitized, session-owned build context."""
    source_root = Path(source).expanduser().resolve()
    destination_root = Path(destination).expanduser().resolve()
    try:
        destination_root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise GeneratedRuntimeError(
            "generated_runtime_destination_inside_source: session artifacts must be outside "
            "the submitted repository"
        )
    plan = detect_generated_runtime(source_root, runtime)
    component = source_root / plan.component if plan.component else source_root
    context = destination_root / "generated-runtime-context"
    if context.exists():
        shutil.rmtree(context)
    context.mkdir(parents=True)
    _copy_sanitized_context(component, context, plan.context_excludes)
    context.joinpath(GENERATED_DOCKERFILE).write_text(plan.dockerfile, encoding="utf-8")
    prepared = GeneratedRuntimePlan(
        **{**asdict(plan), "context_directory": str(context)}
    )
    destination_root.mkdir(parents=True, exist_ok=True)
    destination_root.joinpath(GENERATED_PLAN).write_text(
        json.dumps(
            {
                **asdict(prepared),
                "fingerprint": prepared.fingerprint,
                "generated_files": [
                    f"generated-runtime-context/{GENERATED_DOCKERFILE}",
                    GENERATED_PLAN,
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return prepared


def _runtime_override(runtime: Any | None, **values: str) -> Any:
    payload = {
        "language": str(getattr(runtime, "language", "") or ""),
        "version": str(getattr(runtime, "version", "") or ""),
        "workdir": str(getattr(runtime, "workdir", "") or ""),
        "command": list(getattr(runtime, "command", None) or []),
        "extras": list(getattr(runtime, "extras", None) or []),
        "requirements_files": list(getattr(runtime, "requirements_files", None) or []),
        "context_excludes": list(getattr(runtime, "context_excludes", None) or []),
    }
    payload.update(values)
    return type("RuntimeHints", (), payload)()


def _component_root(root: Path, configured: str) -> Path:
    # Understanding occasionally records ``/`` to mean "the repository root".  A generated
    # runtime can never use the sandbox filesystem root as its source component; its hard
    # boundary is the submitted checkout.  Normalize only this root sentinel while continuing
    # to reject every unrelated absolute path below.
    if configured.strip() == "/":
        return root
    # Contracts produced for a selected monorepo component sometimes retain the
    # repository-relative path with a leading slash (for example
    # ``/examples/drive_thru``).  The submitted ``root`` is already the hard
    # filesystem boundary and the selected component, so an absolute hint whose
    # final component names that root means "the selected root".  Never resolve
    # or read the absolute path itself.
    configured_path = Path(configured) if configured else None
    if (
        configured_path is not None
        and configured_path.is_absolute()
        and configured_path.name == root.name
    ):
        return root
    # The same thing is common without a leading slash: understanding runs at
    # a selected component but records its path relative to the original
    # monorepo (``examples/hotel_receptionist``).  If the submitted root already
    # ends in that exact path, it is provenance for the selected root rather
    # than a request to descend into it a second time.
    if configured_path is not None and not configured_path.is_absolute():
        configured_parts = tuple(part for part in configured_path.parts if part != ".")
        if (
            configured_parts
            and tuple(root.parts[-len(configured_parts) :]) == configured_parts
        ):
            return root
    component = (root / configured).resolve() if configured else root
    try:
        component.relative_to(root)
    except ValueError as exc:
        raise GeneratedRuntimeError(
            "generated_runtime_workdir_escapes_repository"
        ) from exc
    if not component.is_dir():
        raise GeneratedRuntimeError(
            f"generated_runtime_workdir_missing: {configured or '.'}"
        )
    return component


def _python_markers(root: Path) -> list[Path]:
    return [
        root / name
        for name in (
            "pyproject.toml",
            "requirements.txt",
            "uv.lock",
            "poetry.lock",
            "setup.py",
        )
        if root.joinpath(name).is_file()
    ]


def _nested_components(root: Path) -> list[Path]:
    found: set[Path] = set()
    for marker in ("pyproject.toml", "requirements.txt", "setup.py", "package.json"):
        for path in root.glob(f"*/*/{marker}"):
            if not any(
                part in _IGNORED_DIRECTORIES for part in path.relative_to(root).parts
            ):
                found.add(path.parent)
        for path in root.glob(f"*/{marker}"):
            if not any(
                part in _IGNORED_DIRECTORIES for part in path.relative_to(root).parts
            ):
                found.add(path.parent)
    return sorted(found)


def _explicit_command(runtime: Any | None) -> list[str]:
    value = getattr(runtime, "command", None)
    if not value:
        return []
    if isinstance(value, str):
        try:
            command = shlex.split(value)
        except ValueError as exc:
            raise GeneratedRuntimeError("generated_runtime_command_invalid") from exc
    else:
        command = [str(item) for item in value]
    if not command or any(not item or "\x00" in item for item in command):
        raise GeneratedRuntimeError("generated_runtime_command_invalid")
    return command


def _language_version(value: str, default: str) -> str:
    if not value:
        return default
    match = re.search(r"(?<!\d)(\d{1,2}(?:\.\d{1,2})?)(?!\d)", value)
    return match.group(1) if match else default


def _python_language_version(
    root: Path, configured: str, *, interpreter_selector: bool = False
) -> str:
    """Choose an image version or a native interpreter constraint.

    ``>=3.10`` is a compatibility floor, not an instruction to use 3.10. Selecting the first
    number from it makes otherwise reproducible projects fail as transitive dependencies retire
    old interpreter wheels. Exact patch values remain exact; native minor selectors let uv
    resolve a compatible patch. Image ranges prefer the harness' stable default.
    """
    from packaging.markers import InvalidMarker, Marker
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import Version

    declared = configured.strip()
    document = _pyproject_document(root / "pyproject.toml")
    requires = str(document.get("project", {}).get("requires-python", "") or "").strip()
    environments = document.get("tool", {}).get("uv", {}).get("environments", [])
    # shortcut: uv validates platform and patch markers until selection models the target environment.
    python_only = isinstance(environments, list) and all(
        isinstance(item, str)
        and set(re.findall(r"[A-Za-z_]\w*", item)) <= {"python_version", "and", "or", "in", "not"}
        for item in environments
    )
    try:
        markers = [Marker(item) for item in environments] if python_only else []
    except InvalidMarker:
        markers = []

    def supported(version: str) -> bool:
        return not markers or any(
            marker.evaluate({"python_version": ".".join(version.split(".")[:2])})
            for marker in markers
        )
    try:
        project = SpecifierSet(requires)
        if re.fullmatch(r"3\.\d+", declared) and supported(declared):
            if interpreter_selector:
                return str(SpecifierSet(f"=={declared}.*,{requires}".rstrip(",")))
            # Image tags select a moving patch; installation validates Requires-Python.
            return declared
        if declared and not any(marker in declared for marker in "<>=!~,*"):
            if (
                re.fullmatch(r"3\.\d+(?:\.\d+)?", declared)
                and Version(declared) in project
                and supported(declared)
            ):
                return declared
        else:
            specifier = SpecifierSet(declared)
            for candidate in ("3.12", "3.13", "3.11", "3.10", "3.14", "3.9"):
                if Version(candidate) in specifier and Version(candidate) in project and supported(candidate):
                    return candidate
    except (InvalidSpecifier, ValueError):
        pass
    source_pin = ""
    pin_path = root / ".python-version"
    if pin_path.is_file() and pin_path.resolve().is_relative_to(root.resolve()):
        pin = pin_path.read_text(encoding="utf-8", errors="replace").strip()
        if re.fullmatch(r"3\.\d+(?:\.\d+)?", pin):
            source_pin = f"; source .python-version: {pin}"
    raise GeneratedRuntimeError(
        f"generated_runtime_python_version_unsupported: {declared or requires}"
        f" (project requires-python: {requires or 'unspecified'}; uv environments: {environments}{source_pin})"
    )


def _python_install(
    root: Path, runtime: Any | None, command: list[str]
) -> tuple[str, str, list[str], list[str]]:
    selection = _python_extras(root, runtime, command)
    extras = list(selection.names)
    uv_extras = selection.uv_arguments
    uv_script_project = _uv_script_project(root / "pyproject.toml")
    if root.joinpath("uv.lock").is_file() and root.joinpath("pyproject.toml").is_file():
        return (
            "uv.lock",
            "uv-frozen",
            [
                "RUN pip install --no-cache-dir uv==0.8.13",
                "RUN " + shlex.join(["uv", "sync", "--frozen", *uv_extras]),
                'ENV PATH="/app/.venv/bin:$PATH"',
            ],
            extras,
        )
    if root.joinpath("requirements.txt").is_file():
        if selection.all_packages:
            raise GeneratedRuntimeError("generated_runtime_workspace_extras_require_uv")
        if extras and uv_script_project:
            raise GeneratedRuntimeError(
                "generated_runtime_pip_extras_require_installable_project"
            )
        target = ".[" + ",".join(extras) + "]" if extras else None
        return (
            "requirements.txt",
            "pip-requirements",
            [
                "RUN "
                + shlex.join(
                    [
                        "pip",
                        "install",
                        "--no-cache-dir",
                        "-r",
                        "requirements.txt",
                        *([target] if target else []),
                    ]
                )
            ],
            extras,
        )
    if uv_script_project or selection.all_packages:
        install = [
            "RUN pip install --no-cache-dir uv==0.8.13",
            "RUN " + shlex.join(["uv", "sync", "--no-cache", *uv_extras]),
            'ENV PATH="/app/.venv/bin:$PATH"',
        ]
        if _needs_livekit_model_download(root / "pyproject.toml"):
            install.append("RUN python -m livekit.agents download-files")
        return (
            "pyproject.toml",
            "uv-script-project-unlocked",
            install,
            extras,
        )
    if (
        root.joinpath("poetry.lock").is_file()
        and root.joinpath("pyproject.toml").is_file()
    ):
        return (
            "poetry.lock",
            "poetry-locked",
            [
                "RUN pip install --no-cache-dir poetry==2.1.3",
                "RUN poetry config virtualenvs.create false && "
                + shlex.join(
                    [
                        "poetry",
                        "install",
                        "--only",
                        "main",
                        "--no-interaction",
                        *(["--extras", " ".join(extras)] if extras else []),
                    ]
                ),
            ],
            extras,
        )
    if root.joinpath("pyproject.toml").is_file():
        target = ".[" + ",".join(extras) + "]" if extras else "."
        return (
            "pyproject.toml",
            "pip-project",
            ["RUN " + shlex.join(["pip", "install", "--no-cache-dir", target])],
            extras,
        )
    if root.joinpath("setup.py").is_file():
        return (
            "setup.py",
            "pip-setup-project-unlocked",
            ["RUN pip install --no-cache-dir ."],
            [],
        )
    raise GeneratedRuntimeError("generated_runtime_python_dependencies_unsupported")


def _pyproject_document(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import tomllib

        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _uv_script_project(path: Path) -> bool:
    uv = _pyproject_document(path).get("tool", {}).get("uv", {})
    return isinstance(uv, dict) and uv.get("package") is False


def _needs_livekit_model_download(path: Path) -> bool:
    dependencies = _pyproject_document(path).get("project", {}).get("dependencies", [])
    if not isinstance(dependencies, list):
        return False
    normalized = [str(value).lower().replace("_", "-") for value in dependencies]
    return any(value.startswith("livekit-agents") for value in normalized) and any(
        value.startswith(("livekit-plugins-silero", "livekit-plugins-turn-detector"))
        for value in normalized
    )


def python_workspace_members(root: Path) -> set[Path]:
    document = _pyproject_document(root / "pyproject.toml")
    workspace = document.get("tool", {}).get("uv", {}).get("workspace", {})
    members: set[Path] = set()
    excluded: set[Path] = set()
    for key, paths in (("members", members), ("exclude", excluded)):
        for pattern in workspace.get(key, []):
            relative = Path(pattern)
            if relative.is_absolute() or ".." in relative.parts:
                raise GeneratedRuntimeError("generated_runtime_workspace_path_invalid")
            if pattern and not relative.parts:
                continue
            for path in root.glob(pattern):
                resolved = path.resolve()
                if not resolved.is_relative_to(root.resolve()):
                    raise GeneratedRuntimeError(
                        "generated_runtime_workspace_path_invalid"
                    )
                if resolved != root.resolve():
                    paths.add(resolved)
    return members - excluded


def python_workspace_root(source: Path, component: Path) -> Path:
    source, component = source.resolve(), component.resolve()
    if not component.is_relative_to(source):
        raise GeneratedRuntimeError("generated_runtime_workspace_path_invalid")
    for ancestor in component.parents:
        if not ancestor.is_relative_to(source):
            break
        if component in python_workspace_members(ancestor):
            return ancestor
    return component


def select_python_extras(
    root: Path,
    requested: list[str],
    *,
    required_owner: Path | None = None,
    context_excludes: tuple[str, ...] = (),
) -> PythonExtraSelection:
    """Resolve explicit extras in the selected project or unambiguous uv workspace members."""
    if not requested:
        return PythonExtraSelection()
    for name in requested:
        if not re.fullmatch(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*", name):
            raise GeneratedRuntimeError(f"generated_runtime_extra_invalid: {name}")
    names = tuple(sorted({canonicalize_name(name) for name in requested}))
    document = _pyproject_document(root / "pyproject.toml")

    def declared(body: dict[str, Any]) -> set[str]:
        return {
            canonicalize_name(name)
            for name in body.get("project", {}).get("optional-dependencies", {})
        }

    root_names = declared(document)
    if required_owner is None and set(names) <= root_names:
        return PythonExtraSelection(names)
    owners = {name: [root] if name in root_names else [] for name in names}
    for member in sorted(python_workspace_members(root)):
        if _context_path_excluded(
            member.relative_to(root) / "pyproject.toml", context_excludes
        ):
            continue
        available = declared(_pyproject_document(member / "pyproject.toml"))
        for name in names:
            if name in available:
                owners[name].append(member)
    unknown = [name for name, projects in owners.items() if not projects]
    if unknown:
        raise GeneratedRuntimeError(
            "generated_runtime_extra_unknown: " + ", ".join(unknown)
        )
    ambiguous = [name for name, projects in owners.items() if len(projects) > 1]
    if ambiguous:
        raise GeneratedRuntimeError(
            "generated_runtime_extra_ambiguous: " + ", ".join(ambiguous)
        )
    if required_owner is not None and any(
        projects[0].resolve() != required_owner.resolve()
        for projects in owners.values()
    ):
        raise GeneratedRuntimeError("generated_runtime_extra_owner_mismatch")
    return PythonExtraSelection(
        names,
        all_packages=any(
            projects[0].resolve() != root.resolve() for projects in owners.values()
        ),
    )


def supplemental_requirements_command(
    interpreter: str, requirements: tuple[str, ...]
) -> list[str]:
    """Add dependencies without changing the source-selected project resolution."""
    script = (
        "import importlib.metadata as m, subprocess, sys, tempfile; "
        "f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt'); "
        "f.write(''.join(d.metadata['Name'] + '==' + d.version + '\\n' "
        "for d in m.distributions())); f.flush(); "
        "result = subprocess.run(['uv', 'pip', 'install', '--python', sys.executable, "
        "'--constraint', f.name, *[a for p in sys.argv[1:] for a in ('--requirement', p)]]); "
        "f.close(); sys.exit(result.returncode)"
    )
    return [interpreter, "-c", script, *requirements]


def select_python_requirements(
    root: Path, requested: list[str], *, context_excludes: tuple[str, ...] = ()
) -> tuple[str, ...]:
    """Validate selected manifests and their local include graph before packaging."""
    from packaging.requirements import InvalidRequirement, Requirement

    root = root.resolve()
    visited: set[Path] = set()
    active: set[Path] = set()

    def visit(relative: Path) -> None:
        target = root / relative
        if (
            relative.is_absolute()
            or not relative.parts
            or not target.resolve().is_relative_to(root)
            or not target.is_file()
            or any(
                path.is_symlink()
                for path in (target, *target.parents)
                if path != root and path.is_relative_to(root)
            )
            or any(part in _IGNORED_DIRECTORIES for part in relative.parts)
            or _context_path_excluded(relative, context_excludes)
            or _sensitive_file(relative, target)
        ):
            raise GeneratedRuntimeError(
                f"generated_runtime_requirements_path_invalid: {relative}"
            )
        relative = target.resolve().relative_to(root)
        if _context_path_excluded(relative, context_excludes):
            raise GeneratedRuntimeError(
                f"generated_runtime_requirements_path_invalid: {relative}"
            )
        if relative in active:
            raise GeneratedRuntimeError("generated_runtime_requirements_cycle")
        if relative in visited:
            return
        if len(visited) >= 100 or target.stat().st_size > 1024 * 1024:
            raise GeneratedRuntimeError("generated_runtime_requirements_limit")
        visited.add(relative)
        active.add(relative)
        lines = []
        pending = ""
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("#"):
                line = " " + line
            elif line.endswith("\\"):
                pending += line.rstrip("\\")
                continue
            lines.append(pending + line)
            pending = ""
        if pending:
            lines.append(pending)
        for line in lines:
            line = re.split(r"\s+#", line, maxsplit=1)[0].strip()
            if not line or line.startswith("#"):
                continue
            include = re.fullmatch(
                r"(?:-r\s*|--requirement[ =]|-c\s*|--constraint[ =])([^\s]+)", line
            )
            if include:
                visit(relative.parent / include.group(1))
                continue
            hashed = re.fullmatch(
                r"(.+?)(?:\s+--hash(?:=|\s+)(?:sha256:[a-fA-F0-9]{64}"
                r"|sha384:[a-fA-F0-9]{96}|sha512:[a-fA-F0-9]{128}))+",
                line,
            )
            if hashed:
                line = hashed.group(1)
            try:
                requirement = Requirement(line)
            except InvalidRequirement as exc:
                raise GeneratedRuntimeError(
                    f"generated_runtime_requirements_entry_unsupported: {relative}"
                ) from exc
            if requirement.url:
                raise GeneratedRuntimeError(
                    f"generated_runtime_requirements_url_unsupported: {relative}"
                )
        active.remove(relative)

    names = tuple(dict.fromkeys(requested))
    for name in names:
        visit(Path(name))
    return names


def local_editable_install_operands(install: str) -> tuple[tuple[str, list[str]], ...]:
    """Read only complete local editable pip declarations; never execute install text."""
    try:
        tokens = shlex.split(install)
    except ValueError:
        return ()
    if tokens[:2] != ["pip", "install"] or not tokens[2:] or len(tokens[2:]) % 2:
        return ()
    operands = []
    for flag, token in zip(tokens[2::2], tokens[3::2]):
        if flag not in {"-e", "--editable"} or not re.fullmatch(
            r"[A-Za-z0-9_./-]+(?:\[[A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)*\])?", token
        ):
            return ()
        relative, _, groups = token.partition("[")
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            return ()
        operands.append((relative, groups.rstrip("]").split(",") if groups else []))
    return tuple(operands)


def local_install_extra_claims(
    root: Path,
    install: str,
    *,
    reject_unselectable: bool = False,
    context_excludes: tuple[str, ...] = (),
) -> tuple[tuple[Path, str], ...]:
    operands = []
    for relative, groups in local_editable_install_operands(install):
        owner = (root / relative).resolve()
        manifest = owner / "pyproject.toml"
        if (
            not owner.is_relative_to(root.resolve())
            or not manifest.resolve().is_relative_to(root.resolve())
            or not manifest.is_file()
        ):
            return ()
        operands.append((owner, groups))
    claims = []
    for owner, groups in operands:
        declared = (
            _pyproject_document(owner / "pyproject.toml")
            .get("project", {})
            .get("optional-dependencies", {})
        )
        for group in groups:
            if canonicalize_name(group) not in {
                canonicalize_name(name) for name in declared
            }:
                continue
            try:
                selection = select_python_extras(
                    root,
                    [group],
                    required_owner=owner,
                    context_excludes=context_excludes,
                )
            except GeneratedRuntimeError as exc:
                if reject_unselectable and str(exc).startswith(
                    (
                        "generated_runtime_extra_unknown",
                        "generated_runtime_extra_ambiguous",
                        "generated_runtime_extra_owner_mismatch",
                    )
                ):
                    raise
                continue
            claims.append((owner, selection.names[0]))
    return tuple(claims)


def validate_local_install_extras(
    root: Path,
    install: str,
    extras: list[str],
    *,
    original_claims: tuple[tuple[Path, str], ...] = (),
    context_excludes: tuple[str, ...] = (),
) -> None:
    selected = {canonicalize_name(name) for name in extras}
    for owner, name in (
        *original_claims,
        *local_install_extra_claims(
            root, install, reject_unselectable=True, context_excludes=context_excludes
        ),
    ):
        select_python_extras(
            root, [name], required_owner=owner, context_excludes=context_excludes
        )
        if name not in selected:
            raise GeneratedRuntimeError(
                f"generated_runtime_extra_omitted: {owner.relative_to(root.resolve()) / 'pyproject.toml'} requires {name} in runtime.extras"
            )


def _python_extras(
    root: Path, runtime: Any | None, command: list[str]
) -> PythonExtraSelection:
    exclusions = _context_excludes(root, runtime, "pyproject.toml", command)
    requested = [str(item) for item in (getattr(runtime, "extras", None) or [])]
    if requested:
        return select_python_extras(root, requested, context_excludes=exclusions)
    document = _pyproject_document(root / "pyproject.toml")
    declared = document.get("project", {}).get("optional-dependencies", {})
    available = (
        sorted(str(name) for name in declared) if isinstance(declared, dict) else []
    )
    tokens = re.split(r"[^a-z0-9]+", " ".join(command).lower())
    return select_python_extras(
        root,
        [
            extra
            for extra in available
            if extra.lower() not in {"all", "dev", "test", "tests"}
            and extra.lower() in tokens
        ],
        context_excludes=exclusions,
    )


def _python_command(root: Path) -> list[str]:
    scripts = _pyproject_scripts(root / "pyproject.toml")
    if len(scripts) == 1:
        return [scripts[0]]
    if len(scripts) > 1:
        raise GeneratedRuntimeError(
            "generated_runtime_entrypoint_ambiguous: multiple pyproject scripts exist: "
            + ", ".join(scripts)
        )
    candidates = [name for name in _PYTHON_ENTRYPOINTS if root.joinpath(name).is_file()]
    runnable = [
        name
        for name in candidates
        if "__main__"
        in root.joinpath(name).read_text(encoding="utf-8", errors="replace")
        or "cli.run_app"
        in root.joinpath(name).read_text(encoding="utf-8", errors="replace")
    ]
    choices = runnable or candidates
    if len(choices) != 1:
        detail = ", ".join(choices) if choices else "none"
        raise GeneratedRuntimeError(
            "generated_runtime_entrypoint_ambiguous: expected exactly one runnable Python "
            f"entrypoint, found {detail}; set runtime.command"
        )
    command = ["python", choices[0]]
    content = root.joinpath(choices[0]).read_text(encoding="utf-8", errors="replace")
    if "cli.run_app" in content:
        command.append("start")
    return command


def _pyproject_scripts(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        import tomllib

        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    scripts = document.get("project", {}).get("scripts", {})
    return sorted(str(name) for name in scripts) if isinstance(scripts, dict) else []


def _node_install(root: Path) -> tuple[str, str, list[str], str]:
    if root.joinpath("pnpm-lock.yaml").is_file():
        return (
            "pnpm-lock.yaml",
            "pnpm-frozen",
            ["RUN corepack enable && pnpm install --frozen-lockfile"],
            "pnpm",
        )
    if root.joinpath("package-lock.json").is_file():
        return "package-lock.json", "npm-ci", ["RUN npm ci"], "npm"
    if root.joinpath("yarn.lock").is_file():
        return (
            "yarn.lock",
            "yarn-frozen",
            ["RUN corepack enable && yarn install --immutable"],
            "yarn",
        )
    return "package.json", "npm-install-unlocked", ["RUN npm install"], "npm"


def _node_command(root: Path, runner: str) -> list[str]:
    try:
        package = json.loads(root.joinpath("package.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeneratedRuntimeError("generated_runtime_package_json_invalid") from exc
    scripts = package.get("scripts", {})
    if isinstance(scripts, dict) and scripts.get("start"):
        return [runner, "run", "start"] if runner == "npm" else [runner, "start"]
    main = package.get("main")
    if isinstance(main, str) and main and root.joinpath(main).is_file():
        return ["node", main]
    candidates = [name for name in _NODE_ENTRYPOINTS if root.joinpath(name).is_file()]
    if len(candidates) != 1:
        detail = ", ".join(candidates) if candidates else "none"
        raise GeneratedRuntimeError(
            "generated_runtime_entrypoint_ambiguous: package.json needs a start script or "
            f"one existing main file, found {detail}; set runtime.command"
        )
    return ["node", candidates[0]]


def _node_build_steps(root: Path, runner: str, command: list[str]) -> list[str]:
    """Build a submitted Node application when its selected start script needs that phase."""
    try:
        package = json.loads(root.joinpath("package.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    scripts = package.get("scripts", {})
    if not isinstance(scripts, dict) or not scripts.get("build"):
        return []
    selected_start = command in (["npm", "run", "start"], [runner, "start"])
    if not selected_start:
        return []
    invocation = [runner, "run", "build"] if runner == "npm" else [runner, "build"]
    return ["RUN " + shlex.join(invocation)]


def _python_dockerfile(
    version: str,
    dependency: str,
    strategy: str,
    install: list[str],
    command: list[str],
) -> str:
    del dependency, strategy
    return "\n".join(
        [
            f"FROM python:{version}-slim",
            "ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1",
            # Poorly packaged repositories commonly declare Python extensions that
            # only publish wheels for a subset of Python/CPU combinations.  A slim
            # image has no compiler, so otherwise a valid manifest fails depending
            # on the runner architecture (for example webrtcvad on arm64).  This is
            # runtime infrastructure, not a rewrite of the submitted application.
            "RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*",
            "WORKDIR /app",
            "COPY . /app",
            *install,
            "RUN groupadd --gid 10001 alk && useradd --uid 10001 --gid 10001 --no-create-home alk && chown -R alk:alk /app",
            "USER 10001:10001",
            "CMD " + json.dumps(command),
            "",
        ]
    )


def _node_dockerfile(
    version: str, dependency: str, install: list[str], command: list[str]
) -> str:
    del dependency
    return "\n".join(
        [
            f"FROM node:{version}-slim",
            "WORKDIR /app",
            "COPY --chown=node:node . /app",
            *install,
            "USER node",
            "CMD " + json.dumps(command),
            "",
        ]
    )


def _context_excludes(
    component: Path, runtime: Any | None, dependency: str, command: list[str]
) -> tuple[str, ...]:
    required = {
        Path(name)
        for name in (dependency, "pyproject.toml", "package.json")
        if name and component.joinpath(name).is_file()
    }
    for token in command:
        path = Path(token)
        if not token or path.is_absolute() or ".." in path.parts:
            continue
        try:
            if component.joinpath(path).is_file():
                required.add(path)
        except OSError:
            # Inline interpreter programs are arguments, not filesystem paths.
            continue
    values: set[str] = set()
    for raw in getattr(runtime, "context_excludes", None) or []:
        if any(char in str(raw) for char in "?[]"):
            raise GeneratedRuntimeError(
                f"generated_runtime_context_exclude_pattern_unsupported: {raw}"
            )
        relative = Path(str(raw))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise GeneratedRuntimeError(
                f"generated_runtime_context_exclude_invalid: {raw}"
            )
        normalized = relative.as_posix().removeprefix("./")
        matched = list(component.glob(normalized)) if "*" in normalized else [component / normalized]
        matched = [target for target in matched if target.exists() or target.is_symlink()]
        if not matched:
            # Generated/local state files are commonly listed even before the
            # first run creates them. A missing exclusion is already excluded
            # from the immutable build context and is therefore safe to ignore.
            continue
        try:
            for target in matched:
                target.resolve().relative_to(component.resolve())
        except ValueError as exc:
            raise GeneratedRuntimeError(
                f"generated_runtime_context_exclude_invalid: {raw}"
            ) from exc
        if any(_context_path_excluded(path, (normalized,)) for path in required):
            raise GeneratedRuntimeError(
                f"generated_runtime_context_exclude_required: {normalized}"
            )
        values.add(normalized)
    return tuple(sorted(values))


def _context_path_excluded(relative: Path, exclusions: tuple[str, ...]) -> bool:
    def matches(value: str) -> bool:
        return relative.match(value) or (
            value.startswith("**/") and relative.match(value.removeprefix("**/"))
        )

    return any(
        relative == Path(value)
        or Path(value) in relative.parents
        or ("*" in value and matches(value))
        for value in exclusions
    )


def _copy_sanitized_context(
    source: Path, destination: Path, exclusions: tuple[str, ...] = ()
) -> None:
    count = 0
    total = 0
    rejected_files: list[str] = []
    source = source.resolve()

    def copy_file(path: Path, relative: Path) -> None:
        nonlocal count, total
        if _sensitive_file(relative, path):
            return
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise GeneratedRuntimeError(
                f"generated_runtime_file_type_unsupported: {relative.as_posix()}"
            )
        if info.st_size > _MAX_FILE_BYTES:
            raise GeneratedRuntimeError(
                f"generated_runtime_file_too_large: {relative.as_posix()}"
            )
        count += 1
        total += info.st_size
        if count > _MAX_FILES or total > _MAX_CONTEXT_BYTES:
            raise GeneratedRuntimeError("generated_runtime_context_limit_exceeded")
        try:
            inspect_public_file(path)
        except ValueError:
            rejected_files.append(relative.as_posix())
            return
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target, follow_symlinks=True)

    def copy_directory(
        directory: Path, relative_directory: Path, ancestors: frozenset[Path]
    ) -> None:
        resolved_directory = directory.resolve()
        try:
            resolved_directory.relative_to(source)
        except ValueError as exc:
            raise GeneratedRuntimeError(
                f"generated_runtime_symlink_forbidden: {relative_directory.as_posix()}"
            ) from exc
        if resolved_directory in ancestors:
            raise GeneratedRuntimeError(
                f"generated_runtime_symlink_cycle: {relative_directory.as_posix()}"
            )
        next_ancestors = ancestors | {resolved_directory}
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = relative_directory / path.name
            if _context_path_excluded(relative, exclusions):
                continue
            if path.name in _IGNORED_DIRECTORIES and path.is_dir():
                continue
            if path.is_symlink():
                try:
                    resolved = path.resolve(strict=True)
                    resolved.relative_to(source)
                except (OSError, ValueError) as exc:
                    raise GeneratedRuntimeError(
                        f"generated_runtime_symlink_forbidden: {relative.as_posix()}"
                    ) from exc
                if resolved.is_dir():
                    copy_directory(resolved, relative, next_ancestors)
                else:
                    copy_file(resolved, relative)
            elif path.is_dir():
                copy_directory(path, relative, next_ancestors)
            elif path.is_file():
                copy_file(path, relative)
            else:
                raise GeneratedRuntimeError(
                    f"generated_runtime_file_type_unsupported: {relative.as_posix()}"
                )

    copy_directory(source, Path(), frozenset())
    if rejected_files:
        raise GeneratedRuntimeError(
            "generated_runtime_secret_material_detected: "
            + json.dumps({"files": rejected_files[:32], "total": len(rejected_files)})
        )


def _sensitive_file(relative: Path, source: Path) -> bool:
    name = relative.name.lower()
    return (
        (name in _SECRET_NAMES and not is_public_npm_config(source))
        or name.startswith(".env.")
        or (
            relative.suffix.lower() in _SECRET_SUFFIXES
            and not (relative.suffix.lower() == ".pem" and is_public_pem(source))
        )
    )


__all__ = [
    "GENERATED_DOCKERFILE",
    "GENERATED_PLAN",
    "GeneratedRuntimeError",
    "GeneratedRuntimePlan",
    "can_generate_runtime",
    "detect_generated_runtime",
    "prepare_generated_runtime",
]
