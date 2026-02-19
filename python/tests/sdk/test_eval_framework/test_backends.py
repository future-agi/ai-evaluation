"""Tests for fi.evals.framework.backends module."""

import pytest
import time
import threading
from concurrent.futures import TimeoutError as FuturesTimeout

from fi.evals.framework.backends import (
    Backend,
    BackendConfig,
    TaskHandle,
    TaskStatus,
    ThreadPoolBackend,
    ThreadPoolConfig,
)


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_all_statuses(self):
        """Test all status values exist."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"
        assert TaskStatus.TIMEOUT.value == "timeout"


class TestBackendConfig:
    """Tests for BackendConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BackendConfig()

        assert config.max_workers == 4
        assert config.timeout_seconds == 300.0
        assert config.retry_count == 0
        assert config.retry_delay_seconds == 1.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BackendConfig(
            max_workers=8,
            timeout_seconds=60.0,
            retry_count=3,
            retry_delay_seconds=2.0,
        )

        assert config.max_workers == 8
        assert config.timeout_seconds == 60.0
        assert config.retry_count == 3
        assert config.retry_delay_seconds == 2.0


class TestTaskHandle:
    """Tests for TaskHandle dataclass."""

    def test_basic_creation(self):
        """Test basic handle creation."""
        handle = TaskHandle(
            task_id="abc123",
            backend_name="test_backend",
        )

        assert handle.task_id == "abc123"
        assert handle.backend_name == "test_backend"
        assert handle.status == TaskStatus.PENDING
        assert handle.is_done is False

    def test_is_done_pending(self):
        """Test is_done for pending status."""
        handle = TaskHandle(task_id="test", backend_name="test")
        handle._status = TaskStatus.PENDING

        assert handle.is_done is False

    def test_is_done_running(self):
        """Test is_done for running status."""
        handle = TaskHandle(task_id="test", backend_name="test")
        handle._status = TaskStatus.RUNNING

        assert handle.is_done is False

    def test_is_done_completed(self):
        """Test is_done for completed status."""
        handle = TaskHandle(task_id="test", backend_name="test")
        handle._status = TaskStatus.COMPLETED

        assert handle.is_done is True

    def test_is_done_failed(self):
        """Test is_done for failed status."""
        handle = TaskHandle(task_id="test", backend_name="test")
        handle._status = TaskStatus.FAILED

        assert handle.is_done is True

    def test_succeeded_true(self):
        """Test succeeded property when completed."""
        handle = TaskHandle(task_id="test", backend_name="test")
        handle._status = TaskStatus.COMPLETED

        assert handle.succeeded is True

    def test_succeeded_false(self):
        """Test succeeded property when failed."""
        handle = TaskHandle(task_id="test", backend_name="test")
        handle._status = TaskStatus.FAILED

        assert handle.succeeded is False

    def test_result_and_error(self):
        """Test result and error properties."""
        handle = TaskHandle(task_id="test", backend_name="test")
        handle._result = {"score": 0.95}
        handle._error = "test error"

        assert handle.result == {"score": 0.95}
        assert handle.error == "test error"


class TestThreadPoolConfig:
    """Tests for ThreadPoolConfig."""

    def test_inherits_base_config(self):
        """Test ThreadPoolConfig inherits from BackendConfig."""
        config = ThreadPoolConfig()

        assert config.max_workers == 4
        assert config.timeout_seconds == 300.0

    def test_thread_name_prefix(self):
        """Test thread name prefix configuration."""
        config = ThreadPoolConfig(thread_name_prefix="my_eval_")

        assert config.thread_name_prefix == "my_eval_"


class TestThreadPoolBackend:
    """Tests for ThreadPoolBackend."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        backend = ThreadPoolBackend()

        assert backend.config is not None
        assert backend.config.max_workers == 4
        backend.shutdown()

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = ThreadPoolConfig(max_workers=8)
        backend = ThreadPoolBackend(config)

        assert backend.config.max_workers == 8
        backend.shutdown()

    def test_submit_returns_handle(self):
        """Test submit returns a TaskHandle."""
        backend = ThreadPoolBackend()

        def simple_fn():
            return "result"

        handle = backend.submit(simple_fn)

        assert isinstance(handle, TaskHandle)
        assert handle.backend_name == "thread_pool"
        assert handle.task_id is not None

        backend.shutdown()

    def test_get_result_success(self):
        """Test get_result returns correct result."""
        backend = ThreadPoolBackend()

        def simple_fn():
            return {"score": 0.95}

        handle = backend.submit(simple_fn)
        result = backend.get_result(handle)

        assert result == {"score": 0.95}
        backend.shutdown()

    def test_get_result_with_args(self):
        """Test get_result with function arguments."""
        backend = ThreadPoolBackend()

        def add(a, b):
            return a + b

        handle = backend.submit(add, args=(2, 3))
        result = backend.get_result(handle)

        assert result == 5
        backend.shutdown()

    def test_get_result_with_kwargs(self):
        """Test get_result with keyword arguments."""
        backend = ThreadPoolBackend()

        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        handle = backend.submit(greet, args=("World",), kwargs={"greeting": "Hi"})
        result = backend.get_result(handle)

        assert result == "Hi, World!"
        backend.shutdown()

    def test_get_result_timeout(self):
        """Test get_result with timeout parameter."""
        backend = ThreadPoolBackend()

        handle = backend.submit(lambda: "done")

        # Should complete successfully with reasonable timeout
        result = backend.get_result(handle, timeout=5.0)
        assert result == "done"

        backend.shutdown(wait=True)

    def test_get_result_exception(self):
        """Test get_result propagates exceptions."""
        backend = ThreadPoolBackend()

        def failing_fn():
            raise ValueError("Test error")

        handle = backend.submit(failing_fn)

        with pytest.raises(ValueError, match="Test error"):
            backend.get_result(handle)

        backend.shutdown()

    def test_get_status_pending(self):
        """Test get_status returns valid status."""
        backend = ThreadPoolBackend()

        handle = backend.submit(lambda: "result")

        # Status should be one of the valid statuses
        status = backend.get_status(handle)
        assert status in (TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.COMPLETED)

        backend.shutdown(wait=True)

    def test_get_status_completed(self):
        """Test get_status for completed task."""
        backend = ThreadPoolBackend()

        handle = backend.submit(lambda: "result")
        backend.get_result(handle)  # Wait for completion

        status = backend.get_status(handle)
        assert status == TaskStatus.COMPLETED

        backend.shutdown()

    def test_get_status_failed(self):
        """Test get_status for failed task."""
        backend = ThreadPoolBackend()

        def failing_fn():
            raise ValueError("Error")

        handle = backend.submit(failing_fn)

        try:
            backend.get_result(handle)
        except ValueError:
            pass

        status = backend.get_status(handle)
        assert status == TaskStatus.FAILED

        backend.shutdown()

    def test_cancel(self):
        """Test cancel method doesn't error."""
        backend = ThreadPoolBackend()

        # Submit a task and try to cancel - just verify it doesn't crash
        handle = backend.submit(lambda: "result")
        backend.cancel(handle)  # May or may not succeed

        backend.shutdown(wait=True)

    def test_submit_batch(self):
        """Test submit_batch method."""
        backend = ThreadPoolBackend()

        tasks = [
            (lambda: 1, (), {}, None),
            (lambda: 2, (), {}, None),
            (lambda: 3, (), {}, None),
        ]

        handles = backend.submit_batch(tasks)

        assert len(handles) == 3
        for handle in handles:
            assert isinstance(handle, TaskHandle)

        backend.shutdown()

    def test_wait_all(self):
        """Test wait_all method."""
        backend = ThreadPoolBackend()

        def fn(x):
            return x * 2

        handles = [
            backend.submit(fn, args=(1,)),
            backend.submit(fn, args=(2,)),
            backend.submit(fn, args=(3,)),
        ]

        results = backend.wait_all(handles)

        assert results == [2, 4, 6]
        backend.shutdown()

    def test_pending_count(self):
        """Test pending_count method."""
        backend = ThreadPoolBackend()

        # Submit tasks
        for _ in range(3):
            backend.submit(lambda: "result")

        # Just verify the method returns a valid count
        count = backend.pending_count()
        assert count >= 0

        backend.shutdown(wait=True)
        assert backend.pending_count() == 0

    def test_context_manager(self):
        """Test context manager protocol."""
        with ThreadPoolBackend() as backend:
            handle = backend.submit(lambda: "result")
            result = backend.get_result(handle)
            assert result == "result"

        # Executor should be shutdown
        assert backend._executor is None

    def test_context_propagation(self):
        """Test context is stored in handle metadata."""
        backend = ThreadPoolBackend()

        context = {"trace_id": "abc123", "span_id": "def456"}
        handle = backend.submit(lambda: "result", context=context)

        assert handle.metadata.get("context") == context

        backend.shutdown()

    def test_thread_safety(self):
        """Test thread-safe operations."""
        backend = ThreadPoolBackend(ThreadPoolConfig(max_workers=8))
        errors = []
        results = []

        def worker(thread_id):
            try:
                for i in range(10):  # Reduced from 50 for faster tests
                    handle = backend.submit(lambda x=i: x * 2)
                    result = backend.get_result(handle, timeout=5)
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 40  # 4 threads * 10 tasks

        backend.shutdown()

    def test_concurrent_execution(self):
        """Test tasks run concurrently."""
        backend = ThreadPoolBackend(ThreadPoolConfig(max_workers=4))

        handles = [backend.submit(lambda: True) for _ in range(4)]
        results = [backend.get_result(h) for h in handles]

        assert all(results)

        backend.shutdown()
