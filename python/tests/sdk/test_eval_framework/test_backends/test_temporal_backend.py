"""Tests for Temporal backend."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio
import threading

from fi.evals.framework.backends.base import TaskHandle, TaskStatus


class TestTemporalConfig:
    """Tests for TemporalConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        with patch.dict("sys.modules", {"temporalio": MagicMock(), "temporalio.client": MagicMock()}):
            from fi.evals.framework.backends.temporal import TemporalConfig

            config = TemporalConfig()
            assert config.host == "localhost:7233"
            assert config.namespace == "default"
            assert config.task_queue == "eval-tasks"
            assert config.workflow_id_prefix == "eval-"
            assert config.execution_timeout_seconds == 3600.0
            assert config.task_timeout_seconds == 300.0
            assert config.retry_policy_max_attempts == 3
            assert config.retry_policy_initial_interval == 1.0
            assert config.retry_policy_backoff_coefficient == 2.0

    def test_custom_values(self):
        """Test custom configuration values."""
        with patch.dict("sys.modules", {"temporalio": MagicMock(), "temporalio.client": MagicMock()}):
            from fi.evals.framework.backends.temporal import TemporalConfig

            config = TemporalConfig(
                host="temporal.example.com:7233",
                namespace="evaluations",
                task_queue="my-tasks",
                execution_timeout_seconds=7200.0,
            )
            assert config.host == "temporal.example.com:7233"
            assert config.namespace == "evaluations"
            assert config.task_queue == "my-tasks"
            assert config.execution_timeout_seconds == 7200.0


class TestTemporalBackendImport:
    """Tests for Temporal backend import behavior."""

    def test_import_error_without_temporalio(self):
        """Test that import raises ImportError without temporalio."""
        from fi.evals.framework.backends._utils import TEMPORAL

        # Reset the cached state
        TEMPORAL._checked = False
        TEMPORAL._available = False
        TEMPORAL._module = None

        with patch.dict("sys.modules", {"temporalio": None}):
            # Force re-check
            TEMPORAL._checked = False

            # Simulate ImportError
            with patch.object(TEMPORAL, "is_available", return_value=False):
                with pytest.raises(ImportError, match="temporalio"):
                    TEMPORAL.require()


class TestTemporalBackend:
    """Tests for TemporalBackend with mocked Temporal client."""

    @pytest.fixture
    def mock_temporal(self):
        """Create mocked Temporal modules."""
        mock_client_module = MagicMock()
        mock_client = AsyncMock()
        mock_client_module.Client = MagicMock()
        mock_client_module.Client.connect = AsyncMock(return_value=mock_client)
        mock_client_module.WorkflowHandle = MagicMock()
        mock_client_module.WorkflowExecutionStatus = MagicMock()
        mock_client_module.WorkflowExecutionStatus.RUNNING = "RUNNING"
        mock_client_module.WorkflowExecutionStatus.COMPLETED = "COMPLETED"
        mock_client_module.WorkflowExecutionStatus.FAILED = "FAILED"
        mock_client_module.WorkflowExecutionStatus.CANCELED = "CANCELED"
        mock_client_module.WorkflowExecutionStatus.TERMINATED = "TERMINATED"
        mock_client_module.WorkflowExecutionStatus.TIMED_OUT = "TIMED_OUT"

        return {
            "temporalio": MagicMock(),
            "temporalio.client": mock_client_module,
        }

    @pytest.fixture
    def backend(self, mock_temporal):
        """Create a TemporalBackend with mocked dependencies."""
        with patch.dict("sys.modules", mock_temporal):
            from fi.evals.framework.backends._utils import TEMPORAL
            TEMPORAL._checked = True
            TEMPORAL._available = True
            TEMPORAL._module = mock_temporal["temporalio"]

            from fi.evals.framework.backends.temporal import TemporalBackend, TemporalConfig

            config = TemporalConfig(
                host="localhost:7233",
                namespace="test",
            )
            backend = TemporalBackend(config)
            return backend

    def test_backend_name(self, backend):
        """Test backend name is 'temporal'."""
        assert backend.name == "temporal"

    def test_config_stored(self, backend):
        """Test config is stored."""
        assert backend.config.host == "localhost:7233"
        assert backend.config.namespace == "test"

    def test_initial_state(self, backend):
        """Test initial backend state."""
        assert backend._client is None
        assert backend._handles == {}


class TestTemporalBackendSubmit:
    """Tests for TemporalBackend.submit method."""

    @pytest.fixture
    def mock_temporal_full(self):
        """Create fully mocked Temporal setup."""
        mock_workflow_handle = AsyncMock()
        mock_workflow_handle.run_id = "test-run-id"
        mock_workflow_handle.result = AsyncMock(return_value="test-result")
        mock_workflow_handle.describe = AsyncMock()
        mock_workflow_handle.cancel = AsyncMock()

        mock_client = AsyncMock()
        mock_client.start_workflow = AsyncMock(return_value=mock_workflow_handle)
        mock_client.get_workflow_handle = MagicMock(return_value=mock_workflow_handle)

        mock_client_module = MagicMock()
        mock_client_module.Client = MagicMock()
        mock_client_module.Client.connect = AsyncMock(return_value=mock_client)
        mock_client_module.WorkflowHandle = MagicMock()
        mock_client_module.WorkflowExecutionStatus = MagicMock()
        mock_client_module.WorkflowExecutionStatus.RUNNING = "RUNNING"
        mock_client_module.WorkflowExecutionStatus.COMPLETED = "COMPLETED"
        mock_client_module.WorkflowExecutionStatus.FAILED = "FAILED"
        mock_client_module.WorkflowExecutionStatus.CANCELED = "CANCELED"
        mock_client_module.WorkflowExecutionStatus.TERMINATED = "TERMINATED"
        mock_client_module.WorkflowExecutionStatus.TIMED_OUT = "TIMED_OUT"

        return {
            "temporalio": MagicMock(),
            "temporalio.client": mock_client_module,
            "mock_client": mock_client,
            "mock_workflow_handle": mock_workflow_handle,
        }

    def test_submit_creates_handle(self, mock_temporal_full):
        """Test that submit creates a TaskHandle."""
        with patch.dict("sys.modules", {
            "temporalio": mock_temporal_full["temporalio"],
            "temporalio.client": mock_temporal_full["temporalio.client"],
        }):
            from fi.evals.framework.backends._utils import TEMPORAL
            TEMPORAL._checked = True
            TEMPORAL._available = True
            TEMPORAL._module = mock_temporal_full["temporalio"]

            from fi.evals.framework.backends.temporal import TemporalBackend, TemporalConfig

            backend = TemporalBackend(TemporalConfig())

            def test_func(x):
                return x * 2

            # Mock the async operations
            backend._client = mock_temporal_full["mock_client"]
            backend._loop = asyncio.new_event_loop()

            # Run in thread to handle event loop
            def run_submit():
                return backend.submit(test_func, args=(5,))

            with patch.object(backend, "_run_async") as mock_run:
                mock_run.return_value = mock_temporal_full["mock_workflow_handle"]
                handle = backend.submit(test_func, args=(5,))

            assert isinstance(handle, TaskHandle)
            assert handle.backend_name == "temporal"
            assert "test_func" in handle.metadata["function"]

    def test_submit_with_context(self, mock_temporal_full):
        """Test submit with trace context."""
        with patch.dict("sys.modules", {
            "temporalio": mock_temporal_full["temporalio"],
            "temporalio.client": mock_temporal_full["temporalio.client"],
        }):
            from fi.evals.framework.backends._utils import TEMPORAL
            TEMPORAL._checked = True
            TEMPORAL._available = True
            TEMPORAL._module = mock_temporal_full["temporalio"]

            from fi.evals.framework.backends.temporal import TemporalBackend, TemporalConfig

            backend = TemporalBackend(TemporalConfig())
            backend._client = mock_temporal_full["mock_client"]

            context = {"trace_id": "abc123"}

            with patch.object(backend, "_run_async") as mock_run:
                mock_run.return_value = mock_temporal_full["mock_workflow_handle"]
                handle = backend.submit(lambda x: x, args=(1,), context=context)

            assert handle.metadata["context"] == context


class TestTemporalBackendStatus:
    """Tests for TemporalBackend status operations."""

    def test_status_mapping_running(self):
        """Test status mapping for RUNNING."""
        with patch.dict("sys.modules", {"temporalio": MagicMock(), "temporalio.client": MagicMock()}):
            from fi.evals.framework.backends._utils import TEMPORAL
            TEMPORAL._checked = True
            TEMPORAL._available = True

            from fi.evals.framework.backends.temporal import TemporalBackend, TemporalConfig
            from temporalio.client import WorkflowExecutionStatus

            backend = TemporalBackend(TemporalConfig())

            # Mock the status enum
            WorkflowExecutionStatus.RUNNING = "RUNNING"
            status = backend._map_workflow_status(WorkflowExecutionStatus.RUNNING)
            assert status == TaskStatus.RUNNING

    def test_status_mapping_completed(self):
        """Test status mapping for COMPLETED."""
        with patch.dict("sys.modules", {"temporalio": MagicMock(), "temporalio.client": MagicMock()}):
            from fi.evals.framework.backends._utils import TEMPORAL
            TEMPORAL._checked = True
            TEMPORAL._available = True

            from fi.evals.framework.backends.temporal import TemporalBackend, TemporalConfig
            from temporalio.client import WorkflowExecutionStatus

            backend = TemporalBackend(TemporalConfig())

            WorkflowExecutionStatus.COMPLETED = "COMPLETED"
            status = backend._map_workflow_status(WorkflowExecutionStatus.COMPLETED)
            assert status == TaskStatus.COMPLETED

    def test_status_mapping_failed(self):
        """Test status mapping for FAILED."""
        with patch.dict("sys.modules", {"temporalio": MagicMock(), "temporalio.client": MagicMock()}):
            from fi.evals.framework.backends._utils import TEMPORAL
            TEMPORAL._checked = True
            TEMPORAL._available = True

            from fi.evals.framework.backends.temporal import TemporalBackend, TemporalConfig
            from temporalio.client import WorkflowExecutionStatus

            backend = TemporalBackend(TemporalConfig())

            WorkflowExecutionStatus.FAILED = "FAILED"
            status = backend._map_workflow_status(WorkflowExecutionStatus.FAILED)
            assert status == TaskStatus.FAILED


class TestTemporalBackendShutdown:
    """Tests for TemporalBackend shutdown."""

    def test_shutdown_stops_event_loop(self):
        """Test that shutdown stops the event loop."""
        with patch.dict("sys.modules", {"temporalio": MagicMock(), "temporalio.client": MagicMock()}):
            from fi.evals.framework.backends._utils import TEMPORAL
            TEMPORAL._checked = True
            TEMPORAL._available = True

            from fi.evals.framework.backends.temporal import TemporalBackend, TemporalConfig

            backend = TemporalBackend(TemporalConfig())

            # Setup a mock event loop
            mock_loop = MagicMock()
            mock_thread = MagicMock()
            backend._loop = mock_loop
            backend._loop_thread = mock_thread
            backend._client = MagicMock()

            backend.shutdown()

            mock_loop.call_soon_threadsafe.assert_called_once()
            mock_thread.join.assert_called_once()
            assert backend._loop is None
            assert backend._client is None

    def test_shutdown_when_not_initialized(self):
        """Test shutdown when not initialized doesn't error."""
        with patch.dict("sys.modules", {"temporalio": MagicMock(), "temporalio.client": MagicMock()}):
            from fi.evals.framework.backends._utils import TEMPORAL
            TEMPORAL._checked = True
            TEMPORAL._available = True

            from fi.evals.framework.backends.temporal import TemporalBackend, TemporalConfig

            backend = TemporalBackend(TemporalConfig())
            # Should not raise
            backend.shutdown()
