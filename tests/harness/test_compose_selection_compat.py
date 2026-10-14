import pytest

from fi.alk.harness.runtime_selection import RuntimeSelectionError, select_runtime


def select(root, **kwargs):
    return select_runtime(
        root,
        {"execution": "compose", "compose_file": "compose.yaml", "service": "app"},
        validate_environment_dependencies=False,
        **kwargs,
    )


def test_nested_defaults_and_explicit_public_environment(tmp_path):
    (tmp_path / "compose.yaml").write_text(
        'services:\n  app:\n    image: ${APP_IMAGE:-example/app:${TAG:-latest}}\n    environment: ["URL=${SECRET_URL}"]\n'
    )
    assert select(tmp_path).services["app"]["image"] == "example/app:latest"
    result = select(tmp_path, public_environment={"TAG": "pinned"})
    assert result.services["app"]["image"] == "example/app:pinned"
    assert result.services["app"]["environment"] == ["URL=${SECRET_URL}"]
    assert (
        select(tmp_path, public_environment={"APP_IMAGE": "exact:1"}).services["app"][
            "image"
        ]
        == "exact:1"
    )


def test_optional_environment_references_do_not_select_services(tmp_path):
    (tmp_path / "compose.yaml").write_text("""services:
  app:
    image: app:1
    environment: ["CACHE_URL=${CACHE_URL:-http://cache:8000}"]
  cache:
    image: cache:1
    depends_on: [db]
  db:
    image: db:1
  unused:
    image: unused:1
""")
    assert set(select(tmp_path).services) == {"app"}
    with pytest.raises(RuntimeSelectionError, match="compose_dependency_undeclared"):
        select_runtime(tmp_path, {"compose_file": "compose.yaml", "service": "app"})


@pytest.mark.parametrize("declared", ["db", "missing"])
def test_native_selection_keeps_declared_dependencies_and_rejects_missing(
    tmp_path, declared
):
    (tmp_path / "compose.yaml").write_text(f"""services:
  app:
    image: app:1
    depends_on: [cache]
    environment: ["OPTIONAL_URL=http://optional:8000"]
  cache:
    image: cache:1
    depends_on: [{declared}]
  db:
    image: db:1
  optional:
    image: optional:1
""")
    if declared == "missing":
        with pytest.raises(
            RuntimeSelectionError, match="compose_dependency_missing: missing"
        ):
            select(tmp_path)
    else:
        assert list(select(tmp_path).services) == ["db", "cache", "app"]


def test_includes_keep_own_relative_paths_and_resources(tmp_path):
    (tmp_path / "parts").mkdir()
    (tmp_path / "parts" / "nested").mkdir()
    (tmp_path / "compose.yaml").write_text(
        "include: [parts/db.yaml]\nservices:\n  app:\n    image: app:1\n    depends_on: [db]\n"
    )
    child = tmp_path / "parts" / "db.yaml"
    child.write_text("""include: [nested/cache.yaml]
services:
  db:
    build: .
    env_file: config.env
    volumes: ["./data:/data", "dbdata:/database"]
    depends_on: [cache]
volumes:
  dbdata: {}
networks:
  internal: {}
""")
    (tmp_path / "parts" / "nested" / "cache.yaml").write_text(
        "services:\n  cache:\n    build: .\n"
    )
    original = child.read_bytes()
    result = select(tmp_path)
    assert result.services["db"]["build"]["context"] == "./parts"
    assert result.services["cache"]["build"]["context"] == "./parts/nested"
    assert result.services["db"]["env_file"] == ["./parts/config.env"]
    assert result.services["db"]["volumes"] == [
        "./parts/data:/data",
        "dbdata:/database",
    ]
    assert result.document["volumes"] == {"dbdata": {}}
    assert result.document["networks"] == {"internal": {}}
    assert child.read_bytes() == original


@pytest.mark.parametrize(
    "case,error",
    [
        ("cycle", "compose_include_cycle"),
        ("escape", "compose_include_path_invalid"),
        ("unknown", "compose_dependency_missing"),
        ("duplicate", "compose_include_resource_conflict"),
        ("limit", "compose_include_limit"),
    ],
)
def test_invalid_include_graphs_fail(tmp_path, case, error):
    include = "child.yaml"
    child = "services:\n  db:\n    image: db:1\n"
    if case == "cycle":
        child = "include: [compose.yaml]\n" + child
    elif case == "escape":
        include = "../outside.yaml"
    elif case == "unknown":
        child = "services: {}\n"
    elif case == "duplicate":
        child += "  app:\n    image: other:1\n"
    elif case == "limit":
        child += "#" + "x" * (2 * 1024 * 1024)
    (tmp_path / "child.yaml").write_text(child)
    (tmp_path / "compose.yaml").write_text(
        f"include: [{include}]\nservices:\n  app:\n    image: app:1\n    depends_on: [db]\n"
    )
    with pytest.raises(RuntimeSelectionError, match=error):
        select(tmp_path)


def test_included_bind_cannot_escape_source(tmp_path):
    (tmp_path / "compose.yaml").write_text(
        "include: [parts/child.yaml]\nservices: {}\n"
    )
    (tmp_path / "parts").mkdir()
    (tmp_path / "parts" / "child.yaml").write_text(
        'services:\n  app:\n    image: app:1\n    volumes: ["../../outside:/data"]\n'
    )
    with pytest.raises(RuntimeSelectionError, match="compose_source_path_escape"):
        select(tmp_path)


def test_included_build_path_and_args_are_interpolated_once(tmp_path):
    (tmp_path / "parts" / "$data").mkdir(parents=True)
    (tmp_path / "compose.yaml").write_text(
        "include: [parts/child.yaml]\nservices: {}\n"
    )
    (tmp_path / "parts" / "child.yaml").write_text(
        'services:\n  app:\n    build:\n      context: ./$$data\n      args: {LITERAL: "$$HOME", PUBLIC: "${VALUE}"}\n'
    )
    result = select(tmp_path, public_environment={"VALUE": "$untouched"})
    assert result.services["app"]["build"] == {
        "context": "./parts/$data",
        "args": {"LITERAL": "$HOME", "PUBLIC": "$untouched"},
    }
