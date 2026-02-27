"""Tests for Celery backend."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import uuid

from fi.evals.framework.backends.base import TaskHandle, TaskStatus


class TestCeleryConfig:
    """Tests for CeleryConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        with patch.dict("sys.modules", {"celery": MagicMock(), "celery.result": MagicMock()}):
            from fi.evals.framework.backends.celery_backend import CeleryConfig

            config = CeleryConfig()
            assert config.broker_url == "redis://localhost:6379/0"
            assert config.result_backend == "redis://localhost:6379/1"
            assert config.task_queue == "eval_tasks"
            assert config.task_priority == 0
            assert config.task_serializer == "cloudpickle"
            assert config.result_serializer == "cloudpickle"
            assert config.task_acks_late is True
            assert config.task_reject_on_worker_lost is True
            assert config.task_track_started is True
            assert config.worker_prefetch_multiplier == 1

    def test_custom_values(self):
        """Test custom configuration values."""
        with patch.dict("sys.modules", {"celery": MagicMock(), "celery.result": MagicMock()}):
            from fi.evals.framework.backends.celery_backend import CeleryConfig

            config = CeleryConfig(
                broker_url="redis://redis.example.com:6379/0",
                result_backend="redis://redis.example.com:6379/1",
                task_queue="custom_queue",
                task_priority=5,
                task_serializer="json",
            )
            assert config.broker_url == "redis://redis.example.com:6379/0"
            assert config.task_queue == "custom_queue"
            assert config.task_priority == 5
            assert config.task_serializer == "json"


class TestCeleryBackendImport:
    """Tests for Celery backend import behavior."""

    def test_import_error_without_celery(self):
        """Test that import raises ImportError without celery."""
        from fi.evals.framework.backends._utils import CELERY

        # Reset the cached state
        CELERY._checked = False
        CELERY._available = False
        CELERY._module = None

        with patch.object(CELERY, "is_available", return_value=False):
            with pytest.raises(ImportError, match="celery"):
                CELERY.require()


class TestCeleryBackend:
    """Tests for CeleryBackend with mocked Celery."""

    @pytest.fixture
    def mock_celery(self):
        """Create mocked Celery modules."""
        mock_celery_module = MagicMock()
        mock_app = MagicMock()
        mock_celery_module.Celery = MagicMock(return_value=mock_app)

        mock_async_result = MagicMock()
        mock_async_result.get = MagicMock(return_value="result")
        mock_async_result.status = "SUCCESS"
        mock_async_result.revoke = MagicMock()

        mock_task = MagicMock()
        mock_task.apply_async = MagicMock(return_value=mock_async_result)
        mock_task.signature = MagicMock()

        mock_app.task = MagicMock(return_value=lambda f: mock_task)
        mock_app.conf = MagicMock()
        mock_app.control = MagicMock()
        mock_app.pool = MagicMock()

        mock_result_module = MagicMock()
        mock_result_module.AsyncResult = MagicMock(return_value=mock_async_result)

        mock_group = MagicMock()

        return {
            "celery": mock_celery_module,
            "celery.result": mock_result_module,
            "mock_app": mock_app,
            "mock_task": mock_task,
            "mock_async_result": mock_async_result,
            "mock_group": mock_group,
        }

    @pytest.fixture
    def backend(self, mock_celery):
        """Create a CeleryBackend with mocked dependencies."""
        with patch.dict("sys.modules", {
            "celery": mock_celery["celery"],
            "celery.result": mock_celery["celery.result"],
        }):
            from fi.evals.framework.backends._utils import CELERY
            CELERY._checked = True
            CELERY._available = True
            CELERY._module = mock_celery["celery"]

            from fi.evals.framework.backends.celery_backend import CeleryBackend, CeleryConfig

            config = CeleryConfig(
                broker_url="redis://localhost:6379/0",
                task_queue="test_queue",
            )

            # Patch the Celery app creation
            with patch("celery.Celery", return_value=mock_celery["mock_app"]):
                backend = CeleryBackend(config)
                backend._task = mock_celery["mock_task"]
                backend._app = mock_celery["mock_app"]
                return backend

    def test_backend_name(self, backend):
        """Test backend name is 'celery'."""
        assert backend.name == "celery"

    def test_config_stored(self, backend):
        """Test config is stored."""
        assert backend.config.broker_url == "redis://localhost:6379/0"
        assert backend.config.task_queue == "test_queue"


class TestCeleryBackendSubmit:
    """Tests for CeleryBackend.submit method."""

    @pytest.fixture
    def mock_celery_full(self):
        """Create fully mocked Celery setup."""
        mock_async_result = MagicMock()
        mock_async_result.get = MagicMock(return_value="test-result")
        mock_async_result.status = "PENDING"
        mock_async_result.revoke = MagicMock()

        mock_task = MagicMock()
        mock_task.apply_async = MagicMock(return_value=mock_async_result)
        mock_task.signature = MagicMock()

        mock_app = MagicMock()
        mock_app.conf = MagicMock()
        mock_app.control = MagicMock()

        mock_celery_module = MagicMock()
        mock_celery_module.Celery = MagicMock(return_value=mock_app)

        mock_result_module = MagicMock()
        mock_result_module.AsyncResult = MagicMock(return_value=mock_async_result)

        return {
            "celery": mock_celery_module,
            "celery.result": mock_result_module,
            "mock_app": mock_app,
            "mock_task": mock_task,
            "mock_async_result": mock_async_result,
        }

    def test_submit_creates_handle(self, mock_celery_full):
        """Test that submit creates a TaskHandle."""
        with patch.dict("sys.modules", {
            "celery": mock_celery_full["celery"],
            "celery.result": mock_celery_full["celery.result"],
        }):
            from fi.evals.framework.backends._utils import CELERY
            CELERY._checked = True
            CELERY._available = True
            CELERY._module = mock_celery_full["celery"]

            from fi.evals.framework.backends.celery_backend import CeleryBackend, CeleryConfig

            with patch("celery.Celery", return_value=mock_celery_full["mock_app"]):
                backend = CeleryBackend(CeleryConfig())
                backend._task = mock_celery_full["mock_task"]

            def test_func(x):
                return x * 2

            handle = backend.submit(test_func, args=(5,))

            assert isinstance(handle, TaskHandle)
            assert handle.backend_name == "celery"
            assert "test_func" in handle.metadata["function"]
            mock_celery_full["mock_task"].apply_async.assert_called_once()

    def test_submit_with_kwargs(self, mock_celery_full):
        """Test submit with keyword arguments."""
        with patch.dict("sys.modules", {
            "celery": mock_celery_full["celery"],
            "celery.result": mock_celery_full["celery.result"],
        }):
            from fi.evals.framework.backends._utils import CELERY
            CELERY._checked = True
            CELERY._available = True
            CELERY._module = mock_celery_full["celery"]

            from fi.evals.framework.backends.celery_backend import CeleryBackend, CeleryConfig

            with patch("celery.Celery", return_value=mock_celery_full["mock_app"]):
                backend = CeleryBackend(CeleryConfig())
                backend._task = mock_celery_full["mock_task"]

            def test_func(x, y=10):
                return x + y

            handle = backend.submit(test_func, args=(5,), kwargs={"y": 20})

            assert isinstance(handle, TaskHandle)
            # Verify apply_async was called with correct arguments
            call_args = mock_celery_full["mock_task"].apply_async.call_args
            assert call_args[1]["args"][1] == (5,)  # args tuple
            assert call_args[1]["args"][2] == {"y": 20}  # kwargs dict

    def test_submit_with_context(self, mock_celery_full):
        """Test submit stores context in metadata."""
        with patch.dict("sys.modules", {
            "celery": mock_celery_full["celery"],
            "celery.result": mock_celery_full["celery.result"],
        }):
            from fi.evals.framework.backends._utils import CELERY
            CELERY._checked = True
            CELERY._available = True
            CELERY._module = mock_celery_full["celery"]

            from fi.evals.framework.backends.celery_backend import CeleryBackend, CeleryConfig

            with patch("celery.Celery", return_value=mock_celery_full["mock_app"]):
                backend = CeleryBackend(CeleryConfig())
                backend._task = mock_celery_full["mock_task"]

            context = {"trace_id": "abc123", "span_id": "def456"}
            handle = backend.submit(lambda x: x, args=(1,), context=context)

            assert handle.metadata["context"] == context


class TestCeleryBackendStatus:
    """Tests for CeleryBackend status mapping."""

    def test_status_mapping_pending(self):
        """Test status mapping for PENDING."""
        with patch.dict("sys.modules", {"celery": MagicMock(), "celery.result": MagicMock()}):
            from fi.evals.framework.backends._utils import CELERY
            CELERY._checked = True
            CELERY._available = True

            from fi.evals.framework.backends.celery_backend import CeleryBackend, CeleryConfig

            with patch("celery.Celery"):
                backend = CeleryBackend(CeleryConfig())

            status = backend._map_celery_status("PENDING")
            assert status == TaskStatus.PENDING

    def test_status_mapping_started(self):
        """Test status mapping for STARTED."""
        with patch.dict("sys.modules", {"celery": MagicMock(), "celery.result": MagicMock()}):
            from fi.evals.framework.backends._utils import CELERY
            CELERY._checked = True
            CELERY._available = True

            from fi.evals.framework.backends.celery_backend import CeleryBackend, CeleryConfig

            with patch("celery.Celery"):
                backend = CeleryBackend(CeleryConfig())

            status = backend._map_celery_status("STARTED")
            assert status == TaskStatus.RUNNING

    def test_status_mapping_success(self):
        """Test status mapping for SUCCESS."""
        with patch.dict("sys.modules", {"celery": MagicMock(), "celery.result": MagicMock()}):
            from fi.evals.framework.backends._utils import CELERY
            CELERY._checked = True
            CELERY._available = True

            from fi.evals.framework.backends.celery_backend import CeleryBackend, CeleryConfig

            with patch("celery.Celery"):
                backend = CeleryBackend(CeleryConfig())

            status = backend._map_celery_status("SUCCESS")
            assert status == TaskStatus.COMPLETED

    def test_status_mapping_failure(self):
        """Test status mapping for FAILURE."""
        with patch.dict("sys.modules", {"celery": MagicMock(), "celery.result": MagicMock()}):
            from fi.evals.framework.backends._utils import CELERY
            CELERY._checked = True
            CELERY._available = True

            from fi.evals.framework.backends.celery_backend import CeleryBackend, CeleryConfig

            with patch("celery.Celery"):
                backend = CeleryBackend(CeleryConfig())

            status = backend._map_celery_status("FAILURE")
            assert status == TaskStatus.FAILED

    def test_status_mapping_revoked(self):
        """Test status mapping for REVOKED."""
        with patch.dict("sys.modules", {"celery": MagicMock(), "celery.result": MagicMock()}):
            from fi.evals.framework.backends._utils import CELERY
            CELERY._checked = True
            CELERY._available = True

            from fi.evals.framework.backends.celery_backend import CeleryBackend, CeleryConfig

            with patch("celery.Celery"):
                backend = CeleryBackend(CeleryConfig())

            status = backend._map_celery_status("REVOKED")
            assert status == TaskStatus.CANCELLED

    def test_status_mapping_retry(self):
        """Test status mapping for RETRY."""
        with patch.dict("sys.modules", {"celery": MagicMock(), "celery.result": MagicMock()}):
            from fi.evals.framework.backends._utils import CELERY
            CELERY._checked = True
            CELERY._available = True

            from fi.evals.framework.backends.celery_backend import CeleryBackend, CeleryConfig

            with patch("celery.Celery"):
                backend = CeleryBackend(CeleryConfig())

            status = backend._map_celery_status("RETRY")
            assert status == TaskStatus.RUNNING

    def test_status_mapping_unknown(self):
        """Test status mapping for unknown status."""
        with patch.dict("sys.modules", {"celery": MagicMock(), "celery.result": MagicMock()}):
            from fi.evals.framework.backends._utils import CELERY
            CELERY._checked = True
            CELERY._available = True

            from fi.evals.framework.backends.celery_backend import CeleryBackend, CeleryConfig

            with patch("celery.Celery"):
                backend = CeleryBackend(CeleryConfig())

            status = backend._map_celery_status("UNKNOWN_STATUS")
            assert status == TaskStatus.PENDING


class TestCeleryBackendCancel:
    """Tests for CeleryBackend.cancel method."""

    def test_cancel_task(self):
        """Test cancelling a task."""
        with patch.dict("sys.modules", {"celery": MagicMock(), "celery.result": MagicMock()}):
            from fi.evals.framework.backends._utils import CELERY
            CELERY._checked = True
            CELERY._available = True

            from fi.evals.framework.backends.celery_backend import CeleryBackend, CeleryConfig

            mock_async_result = MagicMock()
            mock_async_result.revoke = MagicMock()

            with patch("celery.Celery"):
                backend = CeleryBackend(CeleryConfig())

            # Create a handle and store the async result
            handle = TaskHandle(
                task_id="test-task-id",
                backend_name="celery",
            )
            backend._handles[handle.task_id] = handle
            backend._async_results[handle.task_id] = mock_async_result

            result = backend.cancel(handle)

            assert result is True
            mock_async_result.revoke.assert_called_once_with(terminate=True)
            assert handle._status == TaskStatus.CANCELLED


class TestCeleryBackendShutdown:
    """Tests for CeleryBackend.shutdown method."""

    def test_shutdown_revokes_tracked_tasks(self):
        """Test that shutdown revokes only tracked tasks, not all queues."""
        with patch.dict("sys.modules", {"celery": MagicMock(), "celery.result": MagicMock()}):
            from fi.evals.framework.backends._utils import CELERY
            CELERY._checked = True
            CELERY._available = True

            from fi.evals.framework.backends.celery_backend import CeleryBackend, CeleryConfig

            mock_app = MagicMock()

            with patch("celery.Celery", return_value=mock_app):
                backend = CeleryBackend(CeleryConfig())

            # Add some tracked tasks
            backend._async_results["task-1"] = MagicMock()
            backend._async_results["task-2"] = MagicMock()

            backend.shutdown()

            # Should revoke individual tasks, NOT purge all queues
            assert mock_app.control.revoke.call_count == 2
            mock_app.control.purge.assert_not_called()
            assert len(backend._handles) == 0
            assert len(backend._async_results) == 0


class TestCeleryBackendStats:
    """Tests for CeleryBackend.get_stats method."""

    def test_get_stats(self):
        """Test getting backend statistics."""
        with patch.dict("sys.modules", {"celery": MagicMock(), "celery.result": MagicMock()}):
            from fi.evals.framework.backends._utils import CELERY
            CELERY._checked = True
            CELERY._available = True

            from fi.evals.framework.backends.celery_backend import CeleryBackend, CeleryConfig

            with patch("celery.Celery"):
                backend = CeleryBackend(CeleryConfig(
                    task_queue="test_queue",
                    broker_url="redis://localhost:6379/0",
                ))

            # Add some handles with different statuses
            handle1 = TaskHandle(task_id="1", backend_name="celery")
            handle1._status = TaskStatus.PENDING
            handle2 = TaskHandle(task_id="2", backend_name="celery")
            handle2._status = TaskStatus.RUNNING
            handle3 = TaskHandle(task_id="3", backend_name="celery")
            handle3._status = TaskStatus.RUNNING

            backend._handles = {"1": handle1, "2": handle2, "3": handle3}

            stats = backend.get_stats()

            assert stats["queue"] == "test_queue"
            assert stats["broker"] == "redis://localhost:6379/0"
            assert stats["pending_tasks"] == 1
            assert stats["running_tasks"] == 2
