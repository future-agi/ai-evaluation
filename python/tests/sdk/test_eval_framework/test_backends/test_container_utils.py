"""Tests for shared container utilities.

These utilities are designed to be reused by custom container-based backends
(Docker, ECS, Nomad, etc.) — not just the built-in Kubernetes backend.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from fi.evals.framework.backends._container import (
    DEFAULT_IMAGE,
    EVAL_PAYLOAD_ENV,
    RUNNER_COMMAND,
    RUNNER_SCRIPT,
    parse_result_from_logs,
    serialize_task,
)


class TestConstants:
    """Tests for container constants."""

    def test_default_image(self):
        assert DEFAULT_IMAGE == "fi-eval-runner:latest"

    def test_eval_payload_env(self):
        assert EVAL_PAYLOAD_ENV == "EVAL_PAYLOAD"

    def test_runner_command_is_list(self):
        assert isinstance(RUNNER_COMMAND, list)
        assert RUNNER_COMMAND[0] == "python"
        assert RUNNER_COMMAND[1] == "-c"

    def test_runner_script_contains_key_imports(self):
        # The runner script uses cloudpickle (industry-standard serializer
        # used by Kubeflow, Ray, Dask) in trusted eval environments only.
        assert "cloudpickle" in RUNNER_SCRIPT
        assert "EVAL_PAYLOAD" in RUNNER_SCRIPT
        assert "json.dumps" in RUNNER_SCRIPT

    def test_runner_script_prints_success(self):
        assert '"status": "success"' in RUNNER_SCRIPT

    def test_runner_script_prints_error(self):
        assert '"status": "error"' in RUNNER_SCRIPT


class TestSerializeTask:
    """Tests for serialize_task function."""

    def test_returns_base64_string(self):
        mock_cloudpickle = MagicMock()
        mock_cloudpickle.dumps = MagicMock(return_value=b"test-payload")

        with patch.dict("sys.modules", {"cloudpickle": mock_cloudpickle}):
            result = serialize_task(lambda x: x, (1,), {"y": 2})

        assert isinstance(result, str)
        # Should be valid base64
        import base64
        decoded = base64.b64decode(result)
        assert decoded == b"test-payload"

    def test_calls_serializer_with_correct_args(self):
        mock_cloudpickle = MagicMock()
        mock_cloudpickle.dumps = MagicMock(return_value=b"payload")

        def my_func(x):
            return x

        with patch.dict("sys.modules", {"cloudpickle": mock_cloudpickle}):
            serialize_task(my_func, (42,), {"key": "val"})

        mock_cloudpickle.dumps.assert_called_once()
        call_args = mock_cloudpickle.dumps.call_args[0][0]
        assert call_args[0] is my_func
        assert call_args[1] == (42,)
        assert call_args[2] == {"key": "val"}

    def test_default_kwargs(self):
        mock_cloudpickle = MagicMock()
        mock_cloudpickle.dumps = MagicMock(return_value=b"payload")

        with patch.dict("sys.modules", {"cloudpickle": mock_cloudpickle}):
            serialize_task(lambda: None)

        call_args = mock_cloudpickle.dumps.call_args[0][0]
        assert call_args[1] == ()
        assert call_args[2] == {}

    def test_raises_without_cloudpickle(self):
        with patch.dict("sys.modules", {"cloudpickle": None}):
            with pytest.raises((ImportError, ModuleNotFoundError)):
                serialize_task(lambda: None)


class TestParseResultFromLogs:
    """Tests for parse_result_from_logs function."""

    def test_parses_success(self):
        logs = 'debug output\n{"status": "success", "result": 42}\n'
        assert parse_result_from_logs(logs) == 42

    def test_parses_string_result(self):
        logs = '{"status": "success", "result": "hello"}\n'
        assert parse_result_from_logs(logs) == "hello"

    def test_parses_dict_result(self):
        result_obj = {"score": 0.95, "passed": True}
        logs = json.dumps({"status": "success", "result": result_obj}) + "\n"
        assert parse_result_from_logs(logs) == result_obj

    def test_parses_null_result(self):
        logs = '{"status": "success", "result": null}\n'
        assert parse_result_from_logs(logs) is None

    def test_parses_list_result(self):
        logs = '{"status": "success", "result": [1, 2, 3]}\n'
        assert parse_result_from_logs(logs) == [1, 2, 3]

    def test_error_result_raises(self):
        logs = '{"status": "error", "error": "ZeroDivisionError"}\n'
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            parse_result_from_logs(logs)

    def test_empty_logs_raises(self):
        with pytest.raises(RuntimeError, match="Empty container logs"):
            parse_result_from_logs("")

    def test_whitespace_only_raises(self):
        with pytest.raises(RuntimeError, match="Empty container logs"):
            parse_result_from_logs("   \n  \n  ")

    def test_no_json_raises(self):
        logs = "just plain text\nno json here\n"
        with pytest.raises(RuntimeError, match="No JSON result found"):
            parse_result_from_logs(logs)

    def test_ignores_non_result_json(self):
        logs = '{"some": "other json"}\n{"status": "success", "result": 99}\n'
        assert parse_result_from_logs(logs) == 99

    def test_skips_debug_lines(self):
        logs = (
            "Starting evaluation...\n"
            "Loading model...\n"
            "Running inference...\n"
            '{"status": "success", "result": "done"}\n'
        )
        assert parse_result_from_logs(logs) == "done"

    def test_walks_backwards(self):
        """Last JSON line wins, even if earlier lines have JSON."""
        logs = (
            '{"status": "success", "result": "old"}\n'
            "some debug\n"
            '{"status": "success", "result": "new"}\n'
        )
        assert parse_result_from_logs(logs) == "new"


class TestExportsFromInit:
    """Test that container utilities are accessible from backends package."""

    def test_imports_from_backends(self):
        from fi.evals.framework.backends import (
            DEFAULT_IMAGE,
            EVAL_PAYLOAD_ENV,
            RUNNER_COMMAND,
            RUNNER_SCRIPT,
            serialize_task,
            parse_result_from_logs,
        )
        assert DEFAULT_IMAGE == "fi-eval-runner:latest"
        assert callable(serialize_task)
        assert callable(parse_result_from_logs)

    def test_in_all(self):
        from fi.evals.framework.backends import __all__
        assert "serialize_task" in __all__
        assert "parse_result_from_logs" in __all__
        assert "DEFAULT_IMAGE" in __all__
        assert "RUNNER_SCRIPT" in __all__
