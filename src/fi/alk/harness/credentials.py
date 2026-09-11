"""Safe, deterministic credential discovery for submitted agent repositories.

The reasoning model may later explain a requirement, but it never needs to see a resolved
credential.  This module performs the cheap preflight first and emits a structured manifest the
platform can use to ask for missing secrets before a build or provider call starts.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from enum import Enum
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from fi.simulate.runtime.spec import SecretRef

CREDENTIAL_MANIFEST_VERSION = "futureagi.credential-requirements.v1"


class RequirementKind(str, Enum):
    SECRET = "secret"
    CONFIGURATION = "configuration"
    HARNESS_INFRASTRUCTURE = "harness_infrastructure"


class RequirementStatus(str, Enum):
    MISSING = "missing"
    CONFIGURED = "configured"
    OPTIONAL = "optional"
    HARNESS_PROVIDED = "harness_provided"


class CredentialRequirement(BaseModel):
    id: str
    environment_name: str
    provider: str
    purpose: str
    kind: RequirementKind
    required: bool = True
    status: RequirementStatus
    detected_from: list[str] = Field(default_factory=list)
    accepted_secret_types: list[str] = Field(default_factory=list)


class CredentialChoice(BaseModel):
    id: str
    purpose: str
    options: list[list[str]]
    satisfied: bool = False


class CredentialManifest(BaseModel):
    schema_version: str = CREDENTIAL_MANIFEST_VERSION
    source_digest: str
    detected_connectors: list[str] = Field(default_factory=list)
    requirements: list[CredentialRequirement] = Field(default_factory=list)
    credential_choices: list[CredentialChoice] = Field(default_factory=list)
    scanned_files: int = Field(ge=0)
    truncated: bool = False

    @property
    def missing_required(self) -> list[CredentialRequirement]:
        choice_members = {
            name
            for choice in self.credential_choices
            for option in choice.options
            for name in option
        }
        return [
            item
            for item in self.requirements
            if item.required
            and item.status is RequirementStatus.MISSING
            and item.environment_name not in choice_members
        ]

    @property
    def ready(self) -> bool:
        return not self.missing_required and all(
            choice.satisfied for choice in self.credential_choices
        )


_IGNORED_PARTS = {
    ".git",
    ".venv",
    "venv",
    ".tox",
    ".nox",
    "node_modules",
    "test",
    "tests",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
    "vendor",
}
_SOURCE_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".go",
    ".rb",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
}
_CONFIG_NAMES = {
    ".env.example",
    ".env.sample",
    ".env.template",
    "env.example",
    "env.sample",
    "env.template",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
    "pyproject.toml",
    "package.json",
}
_MAX_FILES = 2_000
_MAX_FILE_BYTES = 1_000_000

_ENV_DECLARATION = re.compile(
    r"(?m)^(?:export[ \t]+)?([A-Z][A-Z0-9_]{2,})[ \t]*=[ \t]*([^\r\n#]*)"
)
_ENV_TEMPLATE_NAMES = {
    ".env.example",
    ".env.sample",
    ".env.template",
    "env.example",
    "env.sample",
    "env.template",
}
_COMPOSE_NAMES = {
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
}
_PLACEHOLDER_VALUE = re.compile(
    r"^(?:"
    r"\[[^\]]*\]|<[^>]*>|\{[^}]*\}|"
    r"(?:your|replace|insert|change)[-_ ]?(?:me|this|.*)|"
    r"example|placeholder|todo|xxx+"
    r")$",
    re.IGNORECASE,
)
_COMPOSE_VARIABLE = re.compile(r"\$\{([A-Z][A-Z0-9_]{2,})(?:(:?[-?])([^}]*))?\}")
_PYTHON_REQUIRED = re.compile(r"os\.environ\s*\[\s*['\"]([A-Z][A-Z0-9_]{2,})['\"]\s*\]")
_PYTHON_GETENV = re.compile(
    r"(?:os\.getenv|os\.environ\.get)\s*\(\s*['\"]([A-Z][A-Z0-9_]{2,})['\"](?:\s*,\s*([^\)]+))?"
)
_JS_ENV = re.compile(r"process\.env\.([A-Z][A-Z0-9_]{2,})")

_SECRET_NAME = re.compile(
    r"(?:^|_)(?:API_?KEY|API_?SECRET|AUTHORIZATION|CREDENTIALS?|PASSWORD|PRIVATE_?KEY|SECRET|TOKEN)(?:_|$)"
)
_HARNESS_PROVIDED = {
    "DATABASE_URL",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "REDIS_URL",
    "TOOLS_API_URL",
    "MCP_URL",
}
_NON_USER_CONFIGURATION = {
    "PORT",
    "HOST",
    "LOG_LEVEL",
    "ENVIRONMENT",
    "NODE_ENV",
    "PYTHONPATH",
}

_PROVIDERS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("LIVEKIT_", "ACCEPTANCE_LIVEKIT_"), "livekit", "connect to the voice room"),
    (("DEEPGRAM_",), "deepgram", "speech-to-text"),
    (("CARTESIA_",), "cartesia", "text-to-speech"),
    (("ELEVENLABS_",), "elevenlabs", "text-to-speech"),
    (("OPENAI_",), "openai", "language or realtime model"),
    (("ANTHROPIC_",), "anthropic", "language model"),
    (("GOOGLE_", "VERTEX_", "CLOUD_ML_"), "google_cloud", "model or cloud runtime"),
    (("VAPI_",), "vapi", "hosted voice agent connection"),
    (("RETELL_",), "retell", "hosted voice agent connection"),
    (("TWILIO_",), "twilio", "telephony connection"),
    (("AWS_",), "aws", "cloud service"),
    (("AZURE_",), "azure", "cloud service"),
    (("MONGODB_", "MONGO_"), "mongodb", "database connection"),
    (("PINECONE_",), "pinecone", "vector database"),
)

_CONNECTOR_SIGNALS = {
    "livekit": ("livekit", "@livekit/", "livekit-agents"),
    "vapi": ("vapi", "@vapi-ai"),
    "retell": ("retell", "retell-sdk"),
    "twilio": ("twilio",),
    "pipecat": ("pipecat",),
    "mcp": ("mcp", "modelcontextprotocol"),
    "http": ("fastapi", "flask", "express", "axios", "httpx"),
}

# Some SDKs consume credentials internally, so the submitted source never calls ``getenv``.
# Connector detection must still make those requirements visible before the expensive build or
# worker-start phase. Keep this list to credentials inherent to the connector itself; model and
# optional application integrations continue to be discovered from explicit source declarations.
_CONNECTOR_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "livekit": ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"),
}

_LIVEKIT_PROVIDER_CREDENTIALS = {
    "deepgram": ("DEEPGRAM_API_KEY",),
    "cartesia": ("CARTESIA_API_KEY",),
    "elevenlabs": ("ELEVENLABS_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
}


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _python_sdk_requirements(content: str) -> list[tuple[str, str]]:
    """Discover credentials consumed internally by constructors in submitted Python.

    This is AST-based because model constructors routinely span multiple lines and contain
    nested calls; a regex ending at the first ``)`` silently misses valid ``vertexai=True``
    configurations.  Parsing is read-only and never imports or executes customer code.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    aliases: dict[str, str] = {}
    imports: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for item in node.names:
                local = item.asname or item.name
                imports[local] = f"{module}.{item.name}" if module else item.name
                if module == "livekit.plugins":
                    aliases[local] = item.name
        elif isinstance(node, ast.Import):
            for item in node.names:
                local = item.asname or item.name.split(".")[0]
                imports[local] = item.name
                prefix = "livekit.plugins."
                if item.name.startswith(prefix):
                    aliases[item.asname or item.name] = item.name[len(prefix) :].split(
                        "."
                    )[0]

    found: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        path = _dotted_name(node.func)
        if not path:
            continue
        parts = path.split(".")
        provider = aliases.get(parts[0], parts[-2] if len(parts) > 1 else "")
        constructor = parts[-1]
        imported_path = imports.get(parts[0], parts[0])
        resolved_path = ".".join([imported_path, *parts[1:]])
        if provider in _LIVEKIT_PROVIDER_CREDENTIALS and constructor in {
            "LLM",
            "STT",
            "TTS",
            "RealtimeModel",
        }:
            for name in _LIVEKIT_PROVIDER_CREDENTIALS[provider]:
                found.add((name, f"sdk:livekit.plugins.{provider}"))
        if provider == "google" and constructor in {"LLM", "STT", "TTS"}:
            vertex = next(
                (
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg == "vertexai"
                ),
                None,
            )
            if isinstance(vertex, ast.Constant) and vertex.value is True:
                for name in (
                    "GOOGLE_APPLICATION_CREDENTIALS",
                    "GOOGLE_APPLICATION_CREDENTIALS_JSON",
                    "GOOGLE_CLOUD_PROJECT",
                ):
                    found.add((name, "sdk:livekit.plugins.google.vertex"))
        # LangChain and Google's first-party SDKs resolve Application Default Credentials
        # internally, so customer code commonly reads only project/location and never calls
        # getenv for the credential file itself. Recognize the constructor and explicit Vertex
        # mode without importing or executing submitted code.
        vertex = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "vertexai"),
            None,
        )
        is_explicit_vertex_client = (
            resolved_path == "langchain_google_genai.ChatGoogleGenerativeAI"
            and isinstance(vertex, ast.Constant)
            and vertex.value is True
        ) or (
            resolved_path in {"google.genai.Client", "genai.Client"}
            and isinstance(vertex, ast.Constant)
            and vertex.value is True
        )
        if is_explicit_vertex_client:
            for name in (
                "GOOGLE_APPLICATION_CREDENTIALS",
                "GOOGLE_APPLICATION_CREDENTIALS_JSON",
                "GOOGLE_CLOUD_PROJECT",
            ):
                found.add((name, f"sdk:{resolved_path}.vertex"))
    return sorted(found)


def discover_credentials(
    root: str | Path,
    *,
    secret_refs: dict[str, SecretRef] | None = None,
    provided_environment: Iterable[str] = (),
    scan_paths: Iterable[str | Path] | None = None,
) -> CredentialManifest:
    """Inspect declarations and environment reads without executing submitted code."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"credential_source_missing: {root}")
    configured = {str(name).upper() for name in provided_environment}
    for alias, ref in (secret_refs or {}).items():
        configured.add(str(alias).upper())
        configured.add(str(ref.key).upper())
        configured.add(_identifier(ref.purpose).upper())
    # Hosted process runtime materializes this vault value to a private file and exports the
    # conventional Google variable to the child.  Discovery deals in names only; no credential
    # value is read or persisted here.
    if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in configured:
        configured.add("GOOGLE_APPLICATION_CREDENTIALS")

    findings: dict[str, dict[str, object]] = {}
    connector_hits: set[str] = set()
    scanned = 0
    truncated = False
    digest = hashlib.sha256()

    for path in _candidate_files(root, scan_paths=scan_paths):
        if scanned >= _MAX_FILES:
            truncated = True
            break
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > _MAX_FILE_BYTES:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        scanned += 1
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(hashlib.sha256(content.encode()).digest())
        lowered = content.lower()
        for connector, signals in _CONNECTOR_SIGNALS.items():
            if any(signal in lowered for signal in signals):
                connector_hits.add(connector)

        def record(
            name: str, *, required: bool, declared_default: bool = False
        ) -> None:
            # ALK_* is a reserved control-plane namespace. Provider adapters and the
            # hosted lifecycle legitimately read these values from their process
            # environment, but they are injected at launch and must never be admitted as
            # customer-supplied target configuration.
            if name in _NON_USER_CONFIGURATION or name.startswith("ALK_"):
                return
            item = findings.setdefault(
                name,
                {
                    "required": False,
                    "declared_default": False,
                    "detected_from": set(),
                },
            )
            item["required"] = bool(item["required"]) or required
            item["declared_default"] = (
                bool(item["declared_default"]) or declared_default
            )
            detected = item["detected_from"]
            assert isinstance(detected, set)
            detected.add(relative)

        # An uppercase assignment in source is usually a constant, not an
        # environment declaration. Only env templates use NAME=value syntax.
        if path.name.lower() in _ENV_TEMPLATE_NAMES:
            for match in _ENV_DECLARATION.finditer(content):
                name = match.group(1)
                value = match.group(2).strip().strip("\"'")
                usable_default = bool(value) and not _PLACEHOLDER_VALUE.fullmatch(value)
                record(
                    name,
                    # A blank non-secret setting in an example file documents a knob; it
                    # does not prove that the selected runtime path needs a value. Strict
                    # source reads and Compose's :? operator remain authoritative. Secret
                    # placeholders stay required because SDKs commonly consume them without
                    # an explicit getenv call in customer code.
                    required=(
                        not usable_default and _kind(name) is RequirementKind.SECRET
                    ),
                    declared_default=usable_default,
                )
        if path.name.lower() in _COMPOSE_NAMES:
            for match in _COMPOSE_VARIABLE.finditer(content):
                operator, fallback = match.group(2), (match.group(3) or "").strip()
                # Compose substitutes an unset plain ${NAME} with an empty string. Only its
                # explicit error operators are admission requirements; source-level strict
                # reads can still make the same name required when a built runtime is scanned.
                required = operator in {"?", ":?"}
                record(
                    match.group(1),
                    required=required,
                    declared_default=bool(fallback) and operator not in {"?", ":?"},
                )
        for match in _PYTHON_REQUIRED.finditer(content):
            # ``os.environ[NAME] if os.environ.get(NAME) else default`` is a common guarded
            # access idiom. The indexed read alone looks mandatory, but the same-line guard
            # proves the application has a fallback and should not block job submission.
            line_start = content.rfind("\n", 0, match.start()) + 1
            line_end = content.find("\n", match.end())
            line = content[line_start : line_end if line_end >= 0 else len(content)]
            name = match.group(1)
            guarded = bool(
                re.search(
                    rf"(?:os\.getenv|os\.environ\.get)\s*\(\s*['\"]{re.escape(name)}['\"]",
                    line,
                )
            )
            record(name, required=not guarded, declared_default=guarded)
        for match in _PYTHON_GETENV.finditer(content):
            default = (match.group(2) or "").strip()
            record(
                match.group(1),
                # getenv/get is explicitly nullable. Treat it as optional unless
                # a blank env template or a strict indexed read says otherwise.
                required=False,
                declared_default=bool(default),
            )
        for match in _JS_ENV.finditer(content):
            record(match.group(1), required=True)
        if path.suffix.lower() == ".py":
            for name, origin in _python_sdk_requirements(content):
                record(name, required=True)
                detected = findings[name]["detected_from"]
                assert isinstance(detected, set)
                detected.add(origin)

    for connector in sorted(connector_hits):
        for name in _CONNECTOR_REQUIREMENTS.get(connector, ()):
            item = findings.setdefault(
                name,
                {
                    "required": True,
                    "declared_default": False,
                    "detected_from": set(),
                },
            )
            item["required"] = True
            detected = item["detected_from"]
            assert isinstance(detected, set)
            detected.add(f"connector:{connector}")

    requirements = []
    for name, finding in sorted(findings.items()):
        kind = _kind(name)
        required = bool(finding["required"])
        if kind is RequirementKind.CONFIGURATION and finding["declared_default"]:
            required = False
        if kind is RequirementKind.HARNESS_INFRASTRUCTURE:
            status = RequirementStatus.HARNESS_PROVIDED
        elif name in configured:
            status = RequirementStatus.CONFIGURED
        elif not required:
            status = RequirementStatus.OPTIONAL
        else:
            status = RequirementStatus.MISSING
        provider, purpose = _provider(name)
        requirements.append(
            CredentialRequirement(
                id=_identifier(name),
                environment_name=name,
                provider=provider,
                purpose=purpose,
                kind=kind,
                required=required,
                status=status,
                detected_from=sorted(finding["detected_from"]),
                accepted_secret_types=(
                    [_secret_type(name)] if kind is RequirementKind.SECRET else []
                ),
            )
        )

    choices = _credential_choices(requirements)
    manifest_core = {
        "files": digest.hexdigest(),
        "connectors": sorted(connector_hits),
        "requirements": [item.model_dump(mode="json") for item in requirements],
        "credential_choices": [item.model_dump(mode="json") for item in choices],
    }
    return CredentialManifest(
        source_digest=hashlib.sha256(
            json.dumps(manifest_core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        detected_connectors=sorted(connector_hits),
        requirements=requirements,
        credential_choices=choices,
        scanned_files=scanned,
        truncated=truncated,
    )


def _candidate_files(
    root: Path, *, scan_paths: Iterable[str | Path] | None = None
) -> list[Path]:
    scopes: list[Path] | None = None
    if scan_paths is not None:
        scopes = []
        for value in scan_paths:
            candidate = (root / value).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            if candidate.exists():
                scopes.append(candidate)
    result: list[Path] = []
    for path in sorted(root.rglob("*")):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if any(part in _IGNORED_PARTS for part in relative.parts):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        if scopes is not None and not any(
            path == scope or (scope.is_dir() and scope in path.parents)
            for scope in scopes
        ):
            continue
        if path.name in _CONFIG_NAMES or path.suffix.lower() in _SOURCE_SUFFIXES:
            result.append(path)
    return result


def _kind(name: str) -> RequirementKind:
    if name in _HARNESS_PROVIDED:
        return RequirementKind.HARNESS_INFRASTRUCTURE
    if _SECRET_NAME.search(name):
        return RequirementKind.SECRET
    return RequirementKind.CONFIGURATION


def _provider(name: str) -> tuple[str, str]:
    for prefixes, provider, purpose in _PROVIDERS:
        if name.startswith(prefixes):
            return provider, purpose
    if name in _HARNESS_PROVIDED:
        return "harness", "generated test infrastructure"
    return "agent", "agent runtime configuration"


def _secret_type(name: str) -> str:
    if "PRIVATE_KEY" in name or name.endswith("CREDENTIALS"):
        return "credential_file"
    if "TOKEN" in name:
        return "token"
    if "SECRET" in name or "PASSWORD" in name:
        return "secret"
    return "api_key"


def _credential_choices(
    requirements: list[CredentialRequirement],
) -> list[CredentialChoice]:
    """Describe common provider auth alternatives without reading secret values."""
    by_name = {item.environment_name: item for item in requirements}
    google_options = [
        ["GEMINI_API_KEY"],
        ["GOOGLE_API_KEY"],
        ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT"],
        ["GOOGLE_APPLICATION_CREDENTIALS_JSON", "GOOGLE_CLOUD_PROJECT"],
    ]
    # Repositories often keep more than one model backend behind LLM_PROVIDER. A strict
    # credential read inside one branch (or a standalone bakeoff utility in the same runtime
    # tree) must not make every backend credential mandatory. The selected provider's one
    # complete authentication route is the requirement. Without an explicit provider selector
    # we remain conservative and do not merge independent integrations into one choice.
    if "LLM_PROVIDER" in by_name:
        google_options.extend(
            [option for option in (["OPENAI_API_KEY"], ["AGENTCC_API_KEY"])]
        )
    present_options = [
        option for option in google_options if all(name in by_name for name in option)
    ]
    google_names = {
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_APPLICATION_CREDENTIALS_JSON",
    }
    if len(present_options) < 2 or not any(
        any(name in google_names for name in option) for option in present_options
    ):
        return []
    configured = {
        RequirementStatus.CONFIGURED,
        RequirementStatus.HARNESS_PROVIDED,
    }
    satisfied = any(
        all(by_name[name].status in configured for name in option)
        for option in present_options
    )
    if not satisfied:
        # These variables are optional individually but one complete route is
        # mandatory. Mark the members input-capable; the choice prevents the UI
        # from requiring every alternative.
        for name in {name for option in present_options for name in option}:
            if by_name[name].status not in configured:
                by_name[name].required = True
                by_name[name].status = RequirementStatus.MISSING
    return [
        CredentialChoice(
            id="google_model_auth",
            purpose="authenticate the Google/Gemini model runtime",
            options=present_options,
            satisfied=satisfied,
        )
    ]


def _identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


__all__ = [
    "CREDENTIAL_MANIFEST_VERSION",
    "CredentialManifest",
    "CredentialChoice",
    "CredentialRequirement",
    "RequirementKind",
    "RequirementStatus",
    "discover_credentials",
]
