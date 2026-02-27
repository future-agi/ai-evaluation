"""Tests for Ray backend."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import uuid

from fi.evals.framework.backends.base import TaskHandle, TaskStatus


class TestRayConfig:
    """Tests for RayConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        with patch.dict("sys.modules", {"ray": MagicMock(), "ray.exceptions": MagicMock()}):
            from fi.evals.framework.backends.ray_backend import RayConfig

            config = RayConfig()
            assert config.address == "auto"
            assert config.num_cpus is None
            assert config.num_gpus == 0.0
            assert config.memory is None
            assert config.runtime_env is None
            assert config.namespace is None
            assert config.ignore_reinit_error is True
            assert config.log_to_driver is True
            assert config.max_retries == 3
            assert config.retry_exceptions is True

    def test_custom_values(self):
        """Test custom configuration values."""
        with patch.dict("sys.modules", {"ray": MagicMock(), "ray.exceptions": MagicMock()}):
            from fi.evals.framework.backends.ray_backend import RayConfig

            config = RayConfig(
                address="ray://cluster:10001",
                num_cpus=2.0,
                num_gpus=0.5,
                memory=1024 * 1024 * 1024,  # 1GB
                namespace="evaluations",
                max_retries=5,
            )
            assert config.address == "ray://cluster:10001"
            assert config.num_cpus == 2.0
            assert config.num_gpus == 0.5
            assert config.memory == 1024 * 1024 * 1024
            assert config.namespace == "evaluations"
            assert config.max_retries == 5


class TestRayBackendImport:
    """Tests for Ray backend import behavior."""

    def test_import_error_without_ray(self):
        """Test that import raises ImportError without ray."""
        from fi.evals.framework.backends._utils import RAY

        # Reset the cached state
        RAY._checked = False
        RAY._available = False
        RAY._module = None

        with patch.object(RAY, "is_available", return_value=False):
            with pytest.raises(ImportError, match="ray"):
                RAY.require()


class TestRayBackend:
    """Tests for RayBackend with mocked Ray."""

    @pytest.fixture
    def mock_ray(self):
        """Create mocked Ray modules."""
        mock_ray_module = MagicMock()
        mock_ray_module.init = MagicMock()
        mock_ray_module.shutdown = MagicMock()
        mock_ray_module.remote = MagicMock()
        mock_ray_module.get = MagicMock(return_value="result")
        mock_ray_module.wait = MagicMock(return_value=([], []))
        mock_ray_module.cancel = MagicMock()
        mock_ray_module.cluster_resources = MagicMock(return_value={"CPU": 8, "GPU": 1})
        mock_ray_module.available_resources = MagicMock(return_value={"CPU": 6, "GPU": 1})

        mock_exceptions = MagicMock()
        mock_exceptions.GetTimeoutError = type("GetTimeoutError", (Exception,), {})

        return {
            "ray": mock_ray_module,
            "ray.exceptions": mock_exceptions,
        }

    @pytest.fixture
    def backend(self, mock_ray):
        """Create a RayBackend with mocked dependencies."""
        with patch.dict("sys.modules", mock_ray):
            from fi.evals.framework.backends._utils import RAY
            RAY._checked = True
            RAY._available = True
            RAY._module = mock_ray["ray"]

            from fi.evals.framework.backends.ray_backend import RayBackend, RayConfig

            config = RayConfig(
                address="local",
                num_cpus=1.0,
            )
            backend = RayBackend(config)
            return backend

    def test_backend_name(self, backend):
        """Test backend name is 'ray'."""
        assert backend.name == "ray"

    def test_config_stored(self, backend):
        """Test config is stored."""
        assert backend.config.address == "local"
        assert backend.config.num_cpus == 1.0

    def test_initialization(self, backend):
        """Test Ray is initialized on backend creation."""
        assert backend._initialized is True


class TestRayBackendSubmit:
    """Tests for RayBackend.submit method."""

    @pytest.fixture
    def mock_ray_full(self):
        """Create fully mocked Ray setup."""
        mock_object_ref = MagicMock()
        mock_object_ref.remote = MagicMock(return_value=mock_object_ref)

        mock_remote_fn = MagicMock()
        mock_remote_fn.remote = MagicMock(return_value=mock_object_ref)

        mock_ray_module = MagicMock()
        mock_ray_module.init = MagicMock()
        mock_ray_module.shutdown = MagicMock()
        mock_ray_module.remote = MagicMock(return_value=lambda f: mock_remote_fn)
        mock_ray_module.get = MagicMock(return_value="test-result")
        mock_ray_module.wait = MagicMock(return_value=([mock_object_ref], []))
        mock_ray_module.cancel = MagicMock()
        mock_ray_module.cluster_resources = MagicMock(return_value={"CPU": 8})
        mock_ray_module.available_resources = MagicMock(return_value={"CPU": 6})

        mock_exceptions = MagicMock()
        mock_exceptions.GetTimeoutError = type("GetTimeoutError", (Exception,), {})

        return {
            "ray": mock_ray_module,
            "ray.exceptions": mock_exceptions,
            "mock_object_ref": mock_object_ref,
            "mock_remote_fn": mock_remote_fn,
        }

    def test_submit_creates_handle(self, mock_ray_full):
        """Test that submit creates a TaskHandle."""
        with patch.dict("sys.modules", {
            "ray": mock_ray_full["ray"],
            "ray.exceptions": mock_ray_full["ray.exceptions"],
        }):
            from fi.evals.framework.backends._utils import RAY
            RAY._checked = True
            RAY._available = True
            RAY._module = mock_ray_full["ray"]

            from fi.evals.framework.backends.ray_backend import RayBackend, RayConfig

            backend = RayBackend(RayConfig())

            def test_func(x):
                return x * 2

            handle = backend.submit(test_func, args=(5,))

            assert isinstance(handle, TaskHandle)
            assert handle.backend_name == "ray"
            assert "test_func" in handle.metadata["function"]

    def test_submit_with_kwargs(self, mock_ray_full):
        """Test submit with keyword arguments."""
        with patch.dict("sys.modules", {
            "ray": mock_ray_full["ray"],
            "ray.exceptions": mock_ray_full["ray.exceptions"],
        }):
            from fi.evals.framework.backends._utils import RAY
            RAY._checked = True
            RAY._available = True
            RAY._module = mock_ray_full["ray"]

            from fi.evals.framework.backends.ray_backend import RayBackend, RayConfig

            backend = RayBackend(RayConfig())

            def test_func(x, y=10):
                return x + y

            handle = backend.submit(test_func, args=(5,), kwargs={"y": 20})

            assert isinstance(handle, TaskHandle)

    def test_submit_with_context(self, mock_ray_full):
        """Test submit stores context in metadata."""
        with patch.dict("sys.modules", {
            "ray": mock_ray_full["ray"],
            "ray.exceptions": mock_ray_full["ray.exceptions"],
        }):
            from fi.evals.framework.backends._utils import RAY
            RAY._checked = True
            RAY._available = True
            RAY._module = mock_ray_full["ray"]

            from fi.evals.framework.backends.ray_backend import RayBackend, RayConfig

            backend = RayBackend(RayConfig())

            context = {"trace_id": "abc123"}
            handle = backend.submit(lambda x: x, args=(1,), context=context)

            assert handle.metadata["context"] == context

    def test_submit_with_resources(self, mock_ray_full):
        """Test submit records resource allocation in metadata."""
        with patch.dict("sys.modules", {
            "ray": mock_ray_full["ray"],
            "ray.exceptions": mock_ray_full["ray.exceptions"],
        }):
            from fi.evals.framework.backends._utils import RAY
            RAY._checked = True
            RAY._available = True
            RAY._module = mock_ray_full["ray"]

            from fi.evals.framework.backends.ray_backend import RayBackend, RayConfig

            backend = RayBackend(RayConfig(num_cpus=2.0, num_gpus=0.5))

            handle = backend.submit(lambda x: x, args=(1,))

            assert handle.metadata["num_cpus"] == 2.0
            assert handle.metadata["num_gpus"] == 0.5


class TestRayBackendGetResult:
    """Tests for RayBackend.get_result method."""

    def test_get_result_success(self):
        """Test getting a successful result."""
        mock_object_ref = MagicMock()

        mock_ray_module = MagicMock()
        mock_ray_module.init = MagicMock()
        mock_ray_module.get = MagicMock(return_value="test-result")
        mock_ray_module.remote = MagicMock(return_value=lambda f: MagicMock(remote=MagicMock(return_value=mock_object_ref)))

        mock_exceptions = MagicMock()
        mock_exceptions.GetTimeoutError = type("GetTimeoutError", (Exception,), {})

        with patch.dict("sys.modules", {
            "ray": mock_ray_module,
            "ray.exceptions": mock_exceptions,
        }):
            from fi.evals.framework.backends._utils import RAY
            RAY._checked = True
            RAY._available = True
            RAY._module = mock_ray_module

            from fi.evals.framework.backends.ray_backend import RayBackend, RayConfig

            backend = RayBackend(RayConfig())

            # Create a handle and store the object ref
            handle = TaskHandle(task_id="test-id", backend_name="ray")
            backend._handles[handle.task_id] = handle
            backend._object_refs[handle.task_id] = mock_object_ref

            result = backend.get_result(handle)

            assert result == "test-result"
            assert handle._status == TaskStatus.COMPLETED

    def test_get_result_unknown_task(self):
        """Test getting result for unknown task raises ValueError."""
        mock_ray_module = MagicMock()
        mock_ray_module.init = MagicMock()

        with patch.dict("sys.modules", {
            "ray": mock_ray_module,
            "ray.exceptions": MagicMock(),
        }):
            from fi.evals.framework.backends._utils import RAY
            RAY._checked = True
            RAY._available = True
            RAY._module = mock_ray_module

            from fi.evals.framework.backends.ray_backend import RayBackend, RayConfig

            backend = RayBackend(RayConfig())

            handle = TaskHandle(task_id="unknown-id", backend_name="ray")
            # Don't add to backend._object_refs

            with pytest.raises(ValueError, match="Unknown task"):
                backend.get_result(handle)

    def test_get_result_general_exception(self):
        """Test getting result when task fails with general exception."""
        mock_object_ref = MagicMock()

        # Create proper exception class
        class GetTimeoutError(Exception):
            pass

        # Create the mock ray module with a proper exceptions attribute
        mock_exceptions_module = MagicMock()
        mock_exceptions_module.GetTimeoutError = GetTimeoutError

        mock_ray_module = MagicMock()
        mock_ray_module.init = MagicMock()
        mock_ray_module.get = MagicMock(side_effect=RuntimeError("task failed"))
        mock_ray_module.exceptions = mock_exceptions_module

        with patch.dict("sys.modules", {
            "ray": mock_ray_module,
            "ray.exceptions": mock_exceptions_module,
        }):
            from fi.evals.framework.backends._utils import RAY
            RAY._checked = True
            RAY._available = True
            RAY._module = mock_ray_module

            from fi.evals.framework.backends.ray_backend import RayBackend, RayConfig

            backend = RayBackend(RayConfig())

            handle = TaskHandle(task_id="test-id", backend_name="ray")
            backend._handles[handle.task_id] = handle
            backend._object_refs[handle.task_id] = mock_object_ref

            with pytest.raises(RuntimeError, match="task failed"):
                backend.get_result(handle)

            assert handle._status == TaskStatus.FAILED
            assert "task failed" in handle._error


class TestRayBackendStatus:
    """Tests for RayBackend.get_status method."""

    def test_get_status_completed(self):
        """Test getting status for completed task."""
        mock_object_ref = MagicMock()

        mock_ray_module = MagicMock()
        mock_ray_module.init = MagicMock()
        mock_ray_module.wait = MagicMock(return_value=([mock_object_ref], []))
        mock_ray_module.get = MagicMock(return_value="result")

        with patch.dict("sys.modules", {
            "ray": mock_ray_module,
            "ray.exceptions": MagicMock(),
        }):
            from fi.evals.framework.backends._utils import RAY
            RAY._checked = True
            RAY._available = True
            RAY._module = mock_ray_module

            from fi.evals.framework.backends.ray_backend import RayBackend, RayConfig

            backend = RayBackend(RayConfig())

            handle = TaskHandle(task_id="test-id", backend_name="ray")
            backend._handles[handle.task_id] = handle
            backend._object_refs[handle.task_id] = mock_object_ref

            status = backend.get_status(handle)

            assert status == TaskStatus.COMPLETED

    def test_get_status_running(self):
        """Test getting status for running task."""
        mock_object_ref = MagicMock()

        mock_ray_module = MagicMock()
        mock_ray_module.init = MagicMock()
        mock_ray_module.wait = MagicMock(return_value=([], [mock_object_ref]))

        with patch.dict("sys.modules", {
            "ray": mock_ray_module,
            "ray.exceptions": MagicMock(),
        }):
            from fi.evals.framework.backends._utils import RAY
            RAY._checked = True
            RAY._available = True
            RAY._module = mock_ray_module

            from fi.evals.framework.backends.ray_backend import RayBackend, RayConfig

            backend = RayBackend(RayConfig())

            handle = TaskHandle(task_id="test-id", backend_name="ray")
            handle._status = TaskStatus.RUNNING
            backend._handles[handle.task_id] = handle
            backend._object_refs[handle.task_id] = mock_object_ref

            status = backend.get_status(handle)

            assert status == TaskStatus.RUNNING

    def test_get_status_cached_completed(self):
        """Test getting cached completed status."""
        with patch.dict("sys.modules", {
            "ray": MagicMock(),
            "ray.exceptions": MagicMock(),
        }):
            from fi.evals.framework.backends._utils import RAY
            RAY._checked = True
            RAY._available = True

            from fi.evals.framework.backends.ray_backend import RayBackend, RayConfig

            backend = RayBackend(RayConfig())

            handle = TaskHandle(task_id="test-id", backend_name="ray")
            handle._status = TaskStatus.COMPLETED
            backend._handles[handle.task_id] = handle

            status = backend.get_status(handle)

            assert status == TaskStatus.COMPLETED


class TestRayBackendCancel:
    """Tests for RayBackend.cancel method."""

    def test_cancel_task(self):
        """Test cancelling a task."""
        mock_object_ref = MagicMock()

        mock_ray_module = MagicMock()
        mock_ray_module.init = MagicMock()
        mock_ray_module.cancel = MagicMock()

        with patch.dict("sys.modules", {
            "ray": mock_ray_module,
            "ray.exceptions": MagicMock(),
        }):
            from fi.evals.framework.backends._utils import RAY
            RAY._checked = True
            RAY._available = True
            RAY._module = mock_ray_module

            from fi.evals.framework.backends.ray_backend import RayBackend, RayConfig

            backend = RayBackend(RayConfig())

            handle = TaskHandle(task_id="test-id", backend_name="ray")
            backend._handles[handle.task_id] = handle
            backend._object_refs[handle.task_id] = mock_object_ref

            result = backend.cancel(handle)

            assert result is True
            mock_ray_module.cancel.assert_called_once_with(mock_object_ref, force=True)
            assert handle._status == TaskStatus.CANCELLED

    def test_cancel_unknown_task(self):
        """Test cancelling unknown task returns False."""
        with patch.dict("sys.modules", {
            "ray": MagicMock(),
            "ray.exceptions": MagicMock(),
        }):
            from fi.evals.framework.backends._utils import RAY
            RAY._checked = True
            RAY._available = True

            from fi.evals.framework.backends.ray_backend import RayBackend, RayConfig

            backend = RayBackend(RayConfig())

            handle = TaskHandle(task_id="unknown-id", backend_name="ray")

            result = backend.cancel(handle)

            assert result is False


class TestRayBackendBatch:
    """Tests for RayBackend batch operations."""

    def test_submit_batch(self):
        """Test batch submission."""
        mock_object_ref = MagicMock()
        mock_remote_fn = MagicMock()
        mock_remote_fn.remote = MagicMock(return_value=mock_object_ref)

        mock_ray_module = MagicMock()
        mock_ray_module.init = MagicMock()
        mock_ray_module.remote = MagicMock(return_value=lambda f: mock_remote_fn)

        with patch.dict("sys.modules", {
            "ray": mock_ray_module,
            "ray.exceptions": MagicMock(),
        }):
            from fi.evals.framework.backends._utils import RAY
            RAY._checked = True
            RAY._available = True
            RAY._module = mock_ray_module

            from fi.evals.framework.backends.ray_backend import RayBackend, RayConfig

            backend = RayBackend(RayConfig())

            def func1(x):
                return x * 2

            def func2(x):
                return x + 1

            tasks = [
                (func1, (1,), {}, None),
                (func2, (2,), {}, None),
                (func1, (3,), {"extra": True}, {"trace": "123"}),
            ]

            handles = backend.submit_batch(tasks)

            assert len(handles) == 3
            assert all(isinstance(h, TaskHandle) for h in handles)
            assert all(h.backend_name == "ray" for h in handles)


class TestRayBackendShutdown:
    """Tests for RayBackend.shutdown method."""

    def test_shutdown_calls_ray_shutdown(self):
        """Test that shutdown calls ray.shutdown()."""
        mock_ray_module = MagicMock()
        mock_ray_module.init = MagicMock()
        mock_ray_module.shutdown = MagicMock()

        with patch.dict("sys.modules", {
            "ray": mock_ray_module,
            "ray.exceptions": MagicMock(),
        }):
            from fi.evals.framework.backends._utils import RAY
            RAY._checked = True
            RAY._available = True
            RAY._module = mock_ray_module

            from fi.evals.framework.backends.ray_backend import RayBackend, RayConfig

            backend = RayBackend(RayConfig())
            backend.shutdown()

            mock_ray_module.shutdown.assert_called_once()
            assert backend._initialized is False

    def test_shutdown_with_wait_false_cancels_tasks(self):
        """Test that shutdown with wait=False cancels tasks."""
        mock_object_ref = MagicMock()

        mock_ray_module = MagicMock()
        mock_ray_module.init = MagicMock()
        mock_ray_module.shutdown = MagicMock()
        mock_ray_module.cancel = MagicMock()

        with patch.dict("sys.modules", {
            "ray": mock_ray_module,
            "ray.exceptions": MagicMock(),
        }):
            from fi.evals.framework.backends._utils import RAY
            RAY._checked = True
            RAY._available = True
            RAY._module = mock_ray_module

            from fi.evals.framework.backends.ray_backend import RayBackend, RayConfig

            backend = RayBackend(RayConfig())
            backend._object_refs["task1"] = mock_object_ref

            backend.shutdown(wait=False)

            mock_ray_module.cancel.assert_called_once_with(mock_object_ref, force=True)


class TestRayBackendStats:
    """Tests for RayBackend.get_stats and get_cluster_resources."""

    def test_get_cluster_resources(self):
        """Test getting cluster resources."""
        mock_ray_module = MagicMock()
        mock_ray_module.init = MagicMock()
        mock_ray_module.cluster_resources = MagicMock(return_value={"CPU": 8, "GPU": 2, "memory": 16000000000})
        mock_ray_module.available_resources = MagicMock(return_value={"CPU": 4, "GPU": 1, "memory": 8000000000})

        with patch.dict("sys.modules", {
            "ray": mock_ray_module,
            "ray.exceptions": MagicMock(),
        }):
            from fi.evals.framework.backends._utils import RAY
            RAY._checked = True
            RAY._available = True
            RAY._module = mock_ray_module

            from fi.evals.framework.backends.ray_backend import RayBackend, RayConfig

            backend = RayBackend(RayConfig())

            resources = backend.get_cluster_resources()

            assert resources["total"]["CPU"] == 8
            assert resources["total"]["GPU"] == 2
            assert resources["available"]["CPU"] == 4
            assert resources["used"]["CPU"] == 4
            assert resources["used"]["GPU"] == 1

    def test_get_stats(self):
        """Test getting backend statistics."""
        mock_ray_module = MagicMock()
        mock_ray_module.init = MagicMock()
        mock_ray_module.cluster_resources = MagicMock(return_value={"CPU": 8})
        mock_ray_module.available_resources = MagicMock(return_value={"CPU": 6})

        with patch.dict("sys.modules", {
            "ray": mock_ray_module,
            "ray.exceptions": MagicMock(),
        }):
            from fi.evals.framework.backends._utils import RAY
            RAY._checked = True
            RAY._available = True
            RAY._module = mock_ray_module

            from fi.evals.framework.backends.ray_backend import RayBackend, RayConfig

            backend = RayBackend(RayConfig(address="ray://cluster:10001"))

            # Add some handles with different statuses
            handle1 = TaskHandle(task_id="1", backend_name="ray")
            handle1._status = TaskStatus.PENDING
            handle2 = TaskHandle(task_id="2", backend_name="ray")
            handle2._status = TaskStatus.RUNNING
            handle3 = TaskHandle(task_id="3", backend_name="ray")
            handle3._status = TaskStatus.COMPLETED
            handle4 = TaskHandle(task_id="4", backend_name="ray")
            handle4._status = TaskStatus.FAILED

            backend._handles = {"1": handle1, "2": handle2, "3": handle3, "4": handle4}

            stats = backend.get_stats()

            assert stats["address"] == "ray://cluster:10001"
            assert stats["pending_tasks"] == 1
            assert stats["running_tasks"] == 1
            assert stats["completed_tasks"] == 1
            assert stats["failed_tasks"] == 1
            assert stats["total_tasks"] == 4
            assert "cluster_resources" in stats
