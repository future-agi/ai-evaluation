from pathlib import Path

from fi.alk.harness.packaging import PackagingKind, inspect_packaging
from fi.alk.harness.provision import compose_file


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_valid_root_compose_is_selected(tmp_path: Path) -> None:
    _write(tmp_path, "compose.yml", "services:\n  agent:\n    image: example/agent\n")

    manifest = inspect_packaging(tmp_path)

    assert manifest.ready
    assert manifest.selected_path == "compose.yml"
    assert manifest.selected_kind is PackagingKind.COMPOSE
    assert manifest.agent_runtime_packaged


def test_missing_dockerfile_bind_input_fails_before_build(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "Dockerfile",
        "FROM python:3.12\nRUN --mount=type=bind,source=uv.lock,target=uv.lock true\n",
    )

    manifest = inspect_packaging(tmp_path)

    assert not manifest.ready
    assert manifest.candidates[0].findings[0].code == "dockerfile_build_input_missing"
    assert "uv.lock" in manifest.candidates[0].findings[0].message


def test_all_multisource_dockerfile_copy_inputs_are_checked(tmp_path: Path) -> None:
    _write(tmp_path, "package.json", "{}\n")
    _write(
        tmp_path,
        "Dockerfile",
        "FROM node:24\nCOPY package.json pnpm-lock.yaml ./\n",
    )

    manifest = inspect_packaging(tmp_path)

    assert not manifest.ready
    missing = [
        finding.message
        for finding in manifest.candidates[0].findings
        if finding.code == "dockerfile_build_input_missing"
    ]
    assert missing == ["Dockerfile requires missing build input: pnpm-lock.yaml"]


def test_monorepo_requires_component_selection(tmp_path: Path) -> None:
    _write(tmp_path, "agents/one/Dockerfile", "FROM python:3.12\nCOPY . .\n")
    _write(tmp_path, "agents/two/Dockerfile", "FROM python:3.12\nCOPY . .\n")

    manifest = inspect_packaging(tmp_path)

    assert not manifest.ready
    assert len(manifest.candidates) == 2
    assert "select the agent subdirectory" in manifest.notes[0]


def test_host_mount_is_visible_without_breaking_local_compose(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "local_setup/docker-compose.yml",
        "services:\n  agent:\n    image: example\n    volumes:\n      - $HOME/.aws:/root/.aws:ro\n",
    )

    manifest = inspect_packaging(tmp_path)

    assert manifest.ready
    assert manifest.selected_path == "local_setup/docker-compose.yml"
    finding = manifest.candidates[0].findings[0]
    assert finding.code == "compose_host_bind_mount"
    assert not finding.blocking
    assert compose_file(tmp_path) == tmp_path / "local_setup/docker-compose.yml"


def test_root_dockerfile_wins_over_nested_example_components(tmp_path: Path) -> None:
    _write(tmp_path, "Dockerfile", "FROM python:3.12\nCOPY . .\n")
    _write(tmp_path, "examples/demo/Dockerfile", "FROM python:3.12\nCOPY . .\n")

    manifest = inspect_packaging(tmp_path)

    assert manifest.ready
    assert manifest.selected_path == "Dockerfile"


def test_production_dockerfile_wins_over_development_root_compose(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "Dockerfile", "FROM python:3.12\nCOPY . .\n")
    _write(
        tmp_path,
        "docker-compose.yml",
        "services:\n  dev:\n    build: .\n    tty: true\n    stdin_open: true\n",
    )

    manifest = inspect_packaging(tmp_path)

    assert manifest.ready
    assert manifest.selected_path == "Dockerfile"
    compose = next(
        item for item in manifest.candidates if item.kind is PackagingKind.COMPOSE
    )
    assert any(
        finding.code == "compose_development_configuration"
        for finding in compose.findings
    )


def test_missing_compose_env_file_fails_before_docker_and_names_service(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "docker-compose.yml",
        "services:\n  voice-app:\n    image: example/voice\n    env_file: [.env]\n",
    )

    manifest = inspect_packaging(tmp_path)

    assert not manifest.ready
    candidate = manifest.candidates[0]
    assert candidate.services == ["voice-app"]
    assert candidate.runtime_candidates == ["voice-app"]
    assert candidate.findings[0].code == "compose_env_file_missing"
    assert "voice-app" in candidate.findings[0].message


def test_uploaded_environment_satisfies_missing_compose_env_file(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "docker-compose.yml",
        "services:\n  voice-app:\n    image: example/voice\n    env_file: [.env]\n",
    )

    manifest = inspect_packaging(tmp_path, external_environment=True)

    assert manifest.ready
    assert manifest.selected_path == "docker-compose.yml"
    finding = manifest.candidates[0].findings[0]
    assert finding.code == "compose_env_file_missing"
    assert finding.blocking is False


def test_optional_compose_env_file_does_not_block_admission(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "compose.yml",
        """services:
  chat-app:
    image: example/chat
    env_file:
      - path: .env.local
        required: false
""",
    )

    manifest = inspect_packaging(tmp_path)

    assert manifest.ready
    assert not any(
        finding.code == "compose_env_file_missing"
        for finding in manifest.candidates[0].findings
    )


def test_named_compose_variant_is_discovered(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "docker-compose.middleware.yaml",
        "services:\n  redis:\n    image: redis:7\n",
    )

    manifest = inspect_packaging(tmp_path)

    assert manifest.ready
    assert manifest.selected_path == "docker-compose.middleware.yaml"


def test_compose_override_fragment_is_not_selected_as_runtime(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "docker-compose.ports.yaml",
        "services:\n  qdrant:\n    ports: ['6333:6333']\n",
    )

    manifest = inspect_packaging(tmp_path)

    assert not manifest.ready
    assert manifest.candidates[0].findings[0].code == "compose_override_fragment"


def test_infrastructure_only_compose_is_not_reported_as_an_agent_runtime(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "docker-compose.yml",
        """services:
  postgres:
    image: postgres:17
  redis:
    image: redis:7
  pgadmin:
    image: dpage/pgadmin4
    profiles: [tools]
""",
    )

    manifest = inspect_packaging(tmp_path)

    assert manifest.ready
    assert manifest.selected_kind is PackagingKind.COMPOSE
    assert manifest.candidates[0].runtime_candidates == []
    assert not manifest.agent_runtime_packaged


def test_custom_built_compose_service_counts_as_packaged_runtime(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "compose.yml",
        """services:
  api:
    build: ./backend
  redis:
    image: redis:7
""",
    )

    manifest = inspect_packaging(tmp_path)

    assert manifest.agent_runtime_packaged
    assert manifest.candidates[0].runtime_candidates == ["api"]


def test_dockerfile_dockerignore_is_not_a_packaging_candidate(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "agent/Dockerfile", "FROM python:3.12-slim\n")
    _write(tmp_path, "agent/Dockerfile.dockerignore", ".venv\n")

    manifest = inspect_packaging(tmp_path)

    assert [candidate.path for candidate in manifest.candidates] == ["agent/Dockerfile"]


def test_prebuilt_api_service_counts_as_packaged_runtime(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "compose.yml",
        """services:
  postgres:
    image: postgres:16
  redis:
    image: redis:7
  api:
    image: ghcr.io/example/voice-api:1.0
""",
    )

    manifest = inspect_packaging(tmp_path)

    assert manifest.agent_runtime_packaged
    assert manifest.candidates[0].runtime_candidates == ["api"]
    assert manifest.candidates[0].runtime_source_roots == []


def test_compose_runtime_records_only_its_build_context_and_env_file(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "api/Dockerfile", "FROM python:3.12-slim\n")
    _write(tmp_path, "api/.env.runtime", "OPENAI_API_KEY=\n")
    _write(tmp_path, "optional/unused.py", "import retell\n")
    _write(
        tmp_path,
        "compose.yml",
        """services:
  api:
    build:
      context: ./api
    env_file: ./api/.env.runtime
  redis:
    image: redis:7
""",
    )

    manifest = inspect_packaging(tmp_path)

    compose = next(
        candidate
        for candidate in manifest.candidates
        if candidate.path == "compose.yml"
    )
    assert compose.runtime_source_roots == [
        "api",
        "api/.env.runtime",
    ]
