"""Tests for Kubernetes backend."""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import uuid

from fi.evals.framework.backends.base import Backend, BackendConfig, TaskHandle, TaskStatus


# ---------------------------------------------------------------------------
# Helpers for building mock kubernetes modules
# ---------------------------------------------------------------------------

def _make_mock_k8s_modules():
    """Create mocked kubernetes modules for patching sys.modules."""
    mock_k8s = MagicMock()
    mock_client = MagicMock()
    mock_config = MagicMock()
    mock_rest = MagicMock()

    # ConfigException used by auto-detect fallback
    mock_config.ConfigException = type("ConfigException", (Exception,), {})

    # ApiException used by cancel / log-read error handling
    mock_rest.ApiException = type("ApiException", (Exception,), {})

    mock_k8s.client = mock_client
    mock_k8s.config = mock_config
    mock_client.rest = mock_rest

    return {
        "kubernetes": mock_k8s,
        "kubernetes.client": mock_client,
        "kubernetes.config": mock_config,
        "kubernetes.client.rest": mock_rest,
    }


def _patch_kubernetes(mock_modules=None):
    """Return a context-manager that patches sys.modules + KUBERNETES flag."""
    mods = mock_modules or _make_mock_k8s_modules()

    class _Ctx:
        def __enter__(self_ctx):
            self_ctx._patcher = patch.dict("sys.modules", mods)
            self_ctx._patcher.__enter__()

            from fi.evals.framework.backends._utils import KUBERNETES
            KUBERNETES._checked = True
            KUBERNETES._available = True
            KUBERNETES._module = mods["kubernetes"]

            return mods

        def __exit__(self_ctx, *exc):
            self_ctx._patcher.__exit__(*exc)

    return _Ctx()


# ---------------------------------------------------------------------------
# TestKubernetesConfig
# ---------------------------------------------------------------------------

class TestKubernetesConfig:
    """Tests for KubernetesConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        with _patch_kubernetes():
            from fi.evals.framework.backends.kubernetes_backend import KubernetesConfig

            config = KubernetesConfig()
            assert config.namespace == "default"
            assert config.image == "fi-eval-runner:latest"
            assert config.cpu_request == "500m"
            assert config.cpu_limit == "1"
            assert config.memory_request == "512Mi"
            assert config.memory_limit == "2Gi"
            assert config.job_prefix == "eval-"
            assert config.backoff_limit == 0
            assert config.active_deadline_seconds == 600
            assert config.ttl_seconds_after_finished == 300
            assert config.service_account_name is None
            assert config.image_pull_secrets is None
            assert config.image_pull_policy == "IfNotPresent"
            assert config.kubeconfig_path is None
            assert config.context is None
            assert config.in_cluster is None
            assert config.labels is None
            assert config.annotations is None
            assert config.poll_interval == 2.0
            assert config.log_tail_lines is None

    def test_custom_values(self):
        """Test custom configuration values."""
        with _patch_kubernetes():
            from fi.evals.framework.backends.kubernetes_backend import KubernetesConfig

            config = KubernetesConfig(
                namespace="evaluations",
                image="my-eval:latest",
                cpu_request="1",
                cpu_limit="2",
                memory_request="1Gi",
                memory_limit="4Gi",
                job_prefix="myeval-",
                backoff_limit=2,
                active_deadline_seconds=1200,
                ttl_seconds_after_finished=600,
                service_account_name="eval-sa",
                image_pull_secrets=["my-secret"],
                image_pull_policy="Always",
                kubeconfig_path="/home/user/.kube/config",
                context="my-context",
                in_cluster=False,
                labels={"team": "ml"},
                annotations={"note": "test"},
                poll_interval=5.0,
                log_tail_lines=100,
            )
            assert config.namespace == "evaluations"
            assert config.image == "my-eval:latest"
            assert config.cpu_request == "1"
            assert config.cpu_limit == "2"
            assert config.memory_request == "1Gi"
            assert config.memory_limit == "4Gi"
            assert config.job_prefix == "myeval-"
            assert config.backoff_limit == 2
            assert config.active_deadline_seconds == 1200
            assert config.ttl_seconds_after_finished == 600
            assert config.service_account_name == "eval-sa"
            assert config.image_pull_secrets == ["my-secret"]
            assert config.image_pull_policy == "Always"
            assert config.kubeconfig_path == "/home/user/.kube/config"
            assert config.context == "my-context"
            assert config.in_cluster is False
            assert config.labels == {"team": "ml"}
            assert config.annotations == {"note": "test"}
            assert config.poll_interval == 5.0
            assert config.log_tail_lines == 100

    def test_inherits_backend_config(self):
        """Test that KubernetesConfig inherits from BackendConfig."""
        with _patch_kubernetes():
            from fi.evals.framework.backends.kubernetes_backend import KubernetesConfig

            config = KubernetesConfig()
            assert isinstance(config, BackendConfig)
            # Has base fields
            assert hasattr(config, "max_workers")
            assert hasattr(config, "timeout_seconds")
            assert hasattr(config, "retry_count")

    def test_timeout_from_base(self):
        """Test that timeout_seconds from BackendConfig is accessible."""
        with _patch_kubernetes():
            from fi.evals.framework.backends.kubernetes_backend import KubernetesConfig

            config = KubernetesConfig(timeout_seconds=120.0)
            assert config.timeout_seconds == 120.0


# ---------------------------------------------------------------------------
# TestKubernetesBackendImport
# ---------------------------------------------------------------------------

class TestKubernetesBackendImport:
    """Tests for Kubernetes backend import behavior."""

    def test_import_error_without_kubernetes(self):
        """Test that import raises ImportError without kubernetes."""
        from fi.evals.framework.backends._utils import KUBERNETES

        KUBERNETES._checked = False
        KUBERNETES._available = False
        KUBERNETES._module = None

        with patch.object(KUBERNETES, "is_available", return_value=False):
            with pytest.raises(ImportError, match="kubernetes"):
                KUBERNETES.require()


# ---------------------------------------------------------------------------
# TestKubernetesBackend  (init / config / state)
# ---------------------------------------------------------------------------

class TestKubernetesBackend:
    """Tests for KubernetesBackend with mocked Kubernetes."""

    @pytest.fixture
    def mock_k8s(self):
        return _make_mock_k8s_modules()

    @pytest.fixture
    def backend(self, mock_k8s):
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )

            config = KubernetesConfig(namespace="test-ns")
            backend = KubernetesBackend(config)
            return backend

    def test_backend_name(self, backend):
        assert backend.name == "kubernetes"

    def test_config_stored(self, backend):
        assert backend.config.namespace == "test-ns"

    def test_initial_state(self, backend):
        assert backend._handles == {}
        assert backend._job_names == {}
        assert backend._batch_api is not None
        assert backend._core_api is not None

    def test_inherits_backend(self, backend):
        assert isinstance(backend, Backend)

    def test_config_loading_in_cluster(self, mock_k8s):
        """Test explicit in-cluster config loading."""
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )

            config = KubernetesConfig(in_cluster=True)
            _ = KubernetesBackend(config)
            mock_k8s["kubernetes.config"].load_incluster_config.assert_called()

    def test_config_loading_out_of_cluster(self, mock_k8s):
        """Test explicit out-of-cluster config loading."""
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )

            config = KubernetesConfig(
                in_cluster=False,
                kubeconfig_path="/tmp/kubeconfig",
                context="dev",
            )
            _ = KubernetesBackend(config)
            mock_k8s["kubernetes.config"].load_kube_config.assert_called_with(
                config_file="/tmp/kubeconfig",
                context="dev",
            )

    def test_config_loading_auto_detect_fallback(self, mock_k8s):
        """Test auto-detect falls back to kubeconfig when in-cluster fails."""
        ConfigException = mock_k8s["kubernetes.config"].ConfigException
        mock_k8s["kubernetes.config"].load_incluster_config.side_effect = ConfigException

        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )

            config = KubernetesConfig()  # in_cluster=None -> auto
            _ = KubernetesBackend(config)
            mock_k8s["kubernetes.config"].load_kube_config.assert_called()


# ---------------------------------------------------------------------------
# TestKubernetesBackendJobName
# ---------------------------------------------------------------------------

class TestKubernetesBackendJobName:
    """Tests for _make_job_name method."""

    @pytest.fixture
    def backend(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            return KubernetesBackend(KubernetesConfig())

    def test_dns_compliance(self, backend):
        """Job name must be DNS-1123 label compliant."""
        name = backend._make_job_name("test-id")
        import re
        assert re.match(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$", name), f"Non-compliant: {name}"

    def test_max_length(self, backend):
        """Job name must not exceed 63 characters."""
        name = backend._make_job_name("test-id")
        assert len(name) <= 63

    def test_uses_prefix(self, backend):
        """Job name starts with the configured prefix."""
        name = backend._make_job_name("test-id")
        assert name.startswith("eval-")

    def test_custom_prefix(self):
        """Job name uses a custom prefix."""
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig(job_prefix="myapp-"))
            name = backend._make_job_name("test-id")
            assert name.startswith("myapp-")

    def test_uniqueness(self, backend):
        """Two calls produce different names."""
        name1 = backend._make_job_name("id1")
        name2 = backend._make_job_name("id2")
        assert name1 != name2


# ---------------------------------------------------------------------------
# TestKubernetesBackendSubmit
# ---------------------------------------------------------------------------

class TestKubernetesBackendSubmit:
    """Tests for KubernetesBackend.submit method."""

    def test_submit_creates_handle(self):
        mock_k8s = _make_mock_k8s_modules()
        mock_cloudpickle = MagicMock()
        mock_cloudpickle.dumps = MagicMock(return_value=b"serialized-data")

        with _patch_kubernetes(mock_k8s):
            with patch.dict("sys.modules", {"cloudpickle": mock_cloudpickle}):
                from fi.evals.framework.backends.kubernetes_backend import (
                    KubernetesBackend,
                    KubernetesConfig,
                )
                backend = KubernetesBackend(KubernetesConfig())

                def test_func(x):
                    return x * 2

                handle = backend.submit(test_func, args=(5,))

                assert isinstance(handle, TaskHandle)
                assert handle.backend_name == "kubernetes"
                assert "test_func" in handle.metadata["function"]
                assert handle.metadata["namespace"] == "default"
                assert "job_name" in handle.metadata

    def test_submit_calls_k8s_api(self):
        mock_k8s = _make_mock_k8s_modules()
        mock_cloudpickle = MagicMock()
        mock_cloudpickle.dumps = MagicMock(return_value=b"serialized-data")

        with _patch_kubernetes(mock_k8s):
            with patch.dict("sys.modules", {"cloudpickle": mock_cloudpickle}):
                from fi.evals.framework.backends.kubernetes_backend import (
                    KubernetesBackend,
                    KubernetesConfig,
                )
                backend = KubernetesBackend(KubernetesConfig())

                handle = backend.submit(lambda x: x, args=(1,))

                # create_namespaced_job should have been called
                backend._batch_api.create_namespaced_job.assert_called_once()

    def test_submit_stores_state(self):
        mock_k8s = _make_mock_k8s_modules()
        mock_cloudpickle = MagicMock()
        mock_cloudpickle.dumps = MagicMock(return_value=b"serialized-data")

        with _patch_kubernetes(mock_k8s):
            with patch.dict("sys.modules", {"cloudpickle": mock_cloudpickle}):
                from fi.evals.framework.backends.kubernetes_backend import (
                    KubernetesBackend,
                    KubernetesConfig,
                )
                backend = KubernetesBackend(KubernetesConfig())
                handle = backend.submit(lambda x: x, args=(1,))

                assert handle.task_id in backend._handles
                assert handle.task_id in backend._job_names

    def test_submit_with_kwargs(self):
        mock_k8s = _make_mock_k8s_modules()
        mock_cloudpickle = MagicMock()
        mock_cloudpickle.dumps = MagicMock(return_value=b"serialized-data")

        with _patch_kubernetes(mock_k8s):
            with patch.dict("sys.modules", {"cloudpickle": mock_cloudpickle}):
                from fi.evals.framework.backends.kubernetes_backend import (
                    KubernetesBackend,
                    KubernetesConfig,
                )
                backend = KubernetesBackend(KubernetesConfig())

                def test_func(x, y=10):
                    return x + y

                handle = backend.submit(test_func, args=(5,), kwargs={"y": 20})
                assert isinstance(handle, TaskHandle)

    def test_submit_with_context(self):
        mock_k8s = _make_mock_k8s_modules()
        mock_cloudpickle = MagicMock()
        mock_cloudpickle.dumps = MagicMock(return_value=b"serialized-data")

        with _patch_kubernetes(mock_k8s):
            with patch.dict("sys.modules", {"cloudpickle": mock_cloudpickle}):
                from fi.evals.framework.backends.kubernetes_backend import (
                    KubernetesBackend,
                    KubernetesConfig,
                )
                backend = KubernetesBackend(KubernetesConfig())

                context = {"trace_id": "abc123"}
                handle = backend.submit(lambda x: x, args=(1,), context=context)

                assert handle.metadata["context"] == context

    def test_submit_failure_sets_failed(self):
        mock_k8s = _make_mock_k8s_modules()
        mock_cloudpickle = MagicMock()
        mock_cloudpickle.dumps = MagicMock(side_effect=RuntimeError("serialize error"))

        with _patch_kubernetes(mock_k8s):
            with patch.dict("sys.modules", {"cloudpickle": mock_cloudpickle}):
                from fi.evals.framework.backends.kubernetes_backend import (
                    KubernetesBackend,
                    KubernetesConfig,
                )
                backend = KubernetesBackend(KubernetesConfig())
                handle = backend.submit(lambda x: x, args=(1,))

                assert handle._status == TaskStatus.FAILED
                assert "serialize error" in handle._error

    def test_submit_api_failure(self):
        mock_k8s = _make_mock_k8s_modules()
        mock_cloudpickle = MagicMock()
        mock_cloudpickle.dumps = MagicMock(return_value=b"serialized-data")

        with _patch_kubernetes(mock_k8s):
            with patch.dict("sys.modules", {"cloudpickle": mock_cloudpickle}):
                from fi.evals.framework.backends.kubernetes_backend import (
                    KubernetesBackend,
                    KubernetesConfig,
                )
                backend = KubernetesBackend(KubernetesConfig())
                backend._batch_api.create_namespaced_job.side_effect = RuntimeError("API error")

                handle = backend.submit(lambda x: x, args=(1,))
                assert handle._status == TaskStatus.FAILED
                assert "API error" in handle._error


# ---------------------------------------------------------------------------
# TestKubernetesBackendGetResult
# ---------------------------------------------------------------------------

class TestKubernetesBackendGetResult:
    """Tests for KubernetesBackend.get_result method."""

    def _make_backend_with_job(self, job_status_sequence, log_output):
        """
        Helper: create a backend with a pre-registered job.

        job_status_sequence: list of TaskStatus values returned by _poll_job_status
        log_output: string returned by _read_job_result
        """
        mock_k8s = _make_mock_k8s_modules()

        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )

            backend = KubernetesBackend(KubernetesConfig(poll_interval=0.01))

            handle = TaskHandle(task_id="test-id", backend_name="kubernetes")
            handle._status = TaskStatus.RUNNING
            backend._handles["test-id"] = handle
            backend._job_names["test-id"] = "eval-abc123"

            status_iter = iter(job_status_sequence)
            backend._poll_job_status = MagicMock(side_effect=status_iter)
            backend._read_job_result = MagicMock(return_value=log_output)

            return backend, handle

    def test_get_result_success(self):
        backend, handle = self._make_backend_with_job(
            [TaskStatus.RUNNING, TaskStatus.COMPLETED],
            "the-result",
        )
        result = backend.get_result(handle, timeout=5)
        assert result == "the-result"
        assert handle._status == TaskStatus.COMPLETED

    def test_get_result_timeout(self):
        backend, handle = self._make_backend_with_job(
            # Always running -- will timeout
            [TaskStatus.RUNNING] * 1000,
            None,
        )
        with pytest.raises(TimeoutError, match="timed out"):
            backend.get_result(handle, timeout=0.05)
        assert handle._status == TaskStatus.TIMEOUT

    def test_get_result_failure(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig(poll_interval=0.01))

            handle = TaskHandle(task_id="test-id", backend_name="kubernetes")
            handle._status = TaskStatus.RUNNING
            backend._handles["test-id"] = handle
            backend._job_names["test-id"] = "eval-abc123"

            backend._poll_job_status = MagicMock(return_value=TaskStatus.FAILED)
            backend._read_job_result = MagicMock(side_effect=RuntimeError("task crashed"))

            with pytest.raises(RuntimeError, match="task crashed"):
                backend.get_result(handle, timeout=5)
            assert handle._status == TaskStatus.FAILED

    def test_get_result_unknown_task(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig())
            handle = TaskHandle(task_id="unknown-id", backend_name="kubernetes")

            with pytest.raises(ValueError, match="Unknown task"):
                backend.get_result(handle)


# ---------------------------------------------------------------------------
# TestKubernetesBackendStatus
# ---------------------------------------------------------------------------

class TestKubernetesBackendStatus:
    """Tests for KubernetesBackend.get_status method."""

    @pytest.fixture
    def backend(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            return KubernetesBackend(KubernetesConfig())

    def test_status_completed_cached(self, backend):
        handle = TaskHandle(task_id="t1", backend_name="kubernetes")
        handle._status = TaskStatus.COMPLETED
        backend._handles["t1"] = handle
        backend._job_names["t1"] = "eval-t1"
        assert backend.get_status(handle) == TaskStatus.COMPLETED

    def test_status_failed_cached(self, backend):
        handle = TaskHandle(task_id="t1", backend_name="kubernetes")
        handle._status = TaskStatus.FAILED
        backend._handles["t1"] = handle
        backend._job_names["t1"] = "eval-t1"
        assert backend.get_status(handle) == TaskStatus.FAILED

    def test_status_cancelled_cached(self, backend):
        handle = TaskHandle(task_id="t1", backend_name="kubernetes")
        handle._status = TaskStatus.CANCELLED
        backend._handles["t1"] = handle
        backend._job_names["t1"] = "eval-t1"
        assert backend.get_status(handle) == TaskStatus.CANCELLED

    def test_status_timeout_cached(self, backend):
        handle = TaskHandle(task_id="t1", backend_name="kubernetes")
        handle._status = TaskStatus.TIMEOUT
        backend._handles["t1"] = handle
        backend._job_names["t1"] = "eval-t1"
        assert backend.get_status(handle) == TaskStatus.TIMEOUT

    def test_status_polls_when_running(self, backend):
        handle = TaskHandle(task_id="t1", backend_name="kubernetes")
        handle._status = TaskStatus.RUNNING
        backend._handles["t1"] = handle
        backend._job_names["t1"] = "eval-t1"

        backend._poll_job_status = MagicMock(return_value=TaskStatus.COMPLETED)
        status = backend.get_status(handle)
        assert status == TaskStatus.COMPLETED
        backend._poll_job_status.assert_called_once_with("eval-t1")

    def test_status_unknown_task(self, backend):
        handle = TaskHandle(task_id="unknown", backend_name="kubernetes")
        assert backend.get_status(handle) == TaskStatus.FAILED

    def test_status_poll_exception_returns_failed(self, backend):
        handle = TaskHandle(task_id="t1", backend_name="kubernetes")
        handle._status = TaskStatus.RUNNING
        backend._handles["t1"] = handle
        backend._job_names["t1"] = "eval-t1"

        backend._poll_job_status = MagicMock(side_effect=RuntimeError("API down"))
        assert backend.get_status(handle) == TaskStatus.FAILED


# ---------------------------------------------------------------------------
# TestKubernetesBackendMapJobStatus
# ---------------------------------------------------------------------------

class TestKubernetesBackendMapJobStatus:
    """Tests for _map_job_status method."""

    @pytest.fixture
    def backend(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            return KubernetesBackend(KubernetesConfig())

    def _make_status(self, conditions=None, active=None, succeeded=None, failed=None):
        status = MagicMock()
        status.conditions = conditions
        status.active = active
        status.succeeded = succeeded
        status.failed = failed
        return status

    def _make_condition(self, ctype, cstatus):
        cond = MagicMock()
        cond.type = ctype
        cond.status = cstatus
        return cond

    def test_completed_via_condition(self, backend):
        cond = self._make_condition("Complete", "True")
        status = self._make_status(conditions=[cond])
        assert backend._map_job_status(status) == TaskStatus.COMPLETED

    def test_failed_via_condition(self, backend):
        cond = self._make_condition("Failed", "True")
        status = self._make_status(conditions=[cond])
        assert backend._map_job_status(status) == TaskStatus.FAILED

    def test_running_via_active(self, backend):
        status = self._make_status(active=1)
        assert backend._map_job_status(status) == TaskStatus.RUNNING

    def test_completed_via_succeeded(self, backend):
        status = self._make_status(succeeded=1)
        assert backend._map_job_status(status) == TaskStatus.COMPLETED

    def test_failed_via_failed_count(self, backend):
        status = self._make_status(failed=1)
        assert backend._map_job_status(status) == TaskStatus.FAILED

    def test_pending_when_empty(self, backend):
        status = self._make_status()
        assert backend._map_job_status(status) == TaskStatus.PENDING

    def test_condition_false_ignored(self, backend):
        cond = self._make_condition("Complete", "False")
        status = self._make_status(conditions=[cond], active=1)
        assert backend._map_job_status(status) == TaskStatus.RUNNING


# ---------------------------------------------------------------------------
# TestKubernetesBackendCancel
# ---------------------------------------------------------------------------

class TestKubernetesBackendCancel:
    """Tests for KubernetesBackend.cancel method."""

    def test_cancel_deletes_job(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig())

            handle = TaskHandle(task_id="t1", backend_name="kubernetes")
            handle._status = TaskStatus.RUNNING
            backend._handles["t1"] = handle
            backend._job_names["t1"] = "eval-t1"

            result = backend.cancel(handle)

            assert result is True
            backend._batch_api.delete_namespaced_job.assert_called_once()
            assert handle._status == TaskStatus.CANCELLED

    def test_cancel_sets_status(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig())

            handle = TaskHandle(task_id="t1", backend_name="kubernetes")
            handle._status = TaskStatus.RUNNING
            backend._handles["t1"] = handle
            backend._job_names["t1"] = "eval-t1"

            backend.cancel(handle)
            assert backend._handles["t1"]._status == TaskStatus.CANCELLED

    def test_cancel_unknown_task(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig())

            handle = TaskHandle(task_id="unknown", backend_name="kubernetes")
            result = backend.cancel(handle)
            assert result is False

    def test_cancel_api_exception(self):
        mock_k8s = _make_mock_k8s_modules()
        ApiException = mock_k8s["kubernetes.client.rest"].ApiException

        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig())

            handle = TaskHandle(task_id="t1", backend_name="kubernetes")
            handle._status = TaskStatus.RUNNING
            backend._handles["t1"] = handle
            backend._job_names["t1"] = "eval-t1"

            backend._batch_api.delete_namespaced_job.side_effect = ApiException("not found")

            result = backend.cancel(handle)
            assert result is False


# ---------------------------------------------------------------------------
# TestKubernetesBackendBatch
# ---------------------------------------------------------------------------

class TestKubernetesBackendBatch:
    """Tests for KubernetesBackend batch operations."""

    def test_submit_batch_returns_handles(self):
        mock_k8s = _make_mock_k8s_modules()
        mock_cloudpickle = MagicMock()
        mock_cloudpickle.dumps = MagicMock(return_value=b"serialized-data")

        with _patch_kubernetes(mock_k8s):
            with patch.dict("sys.modules", {"cloudpickle": mock_cloudpickle}):
                from fi.evals.framework.backends.kubernetes_backend import (
                    KubernetesBackend,
                    KubernetesConfig,
                )
                backend = KubernetesBackend(KubernetesConfig())

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
                assert all(h.backend_name == "kubernetes" for h in handles)
                # Should have called create_namespaced_job 3 times
                assert backend._batch_api.create_namespaced_job.call_count == 3

    def test_submit_batch_empty_list(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig())
            handles = backend.submit_batch([])
            assert handles == []


# ---------------------------------------------------------------------------
# TestKubernetesBackendShutdown
# ---------------------------------------------------------------------------

class TestKubernetesBackendShutdown:
    """Tests for KubernetesBackend.shutdown method."""

    def test_shutdown_wait_true(self):
        """shutdown(wait=True) should not cancel tasks."""
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig())

            h = TaskHandle(task_id="t1", backend_name="kubernetes")
            h._status = TaskStatus.RUNNING
            backend._handles["t1"] = h
            backend._job_names["t1"] = "eval-t1"

            backend.shutdown(wait=True)
            # Job should NOT have been deleted
            backend._batch_api.delete_namespaced_job.assert_not_called()

    def test_shutdown_wait_false_cancels(self):
        """shutdown(wait=False) cancels pending/running tasks."""
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig())

            h1 = TaskHandle(task_id="t1", backend_name="kubernetes")
            h1._status = TaskStatus.RUNNING
            h2 = TaskHandle(task_id="t2", backend_name="kubernetes")
            h2._status = TaskStatus.PENDING
            h3 = TaskHandle(task_id="t3", backend_name="kubernetes")
            h3._status = TaskStatus.COMPLETED  # Should NOT be cancelled

            backend._handles = {"t1": h1, "t2": h2, "t3": h3}
            backend._job_names = {"t1": "eval-t1", "t2": "eval-t2", "t3": "eval-t3"}

            backend.shutdown(wait=False)

            # Only pending/running jobs should be deleted
            assert backend._batch_api.delete_namespaced_job.call_count == 2

    def test_shutdown_logs_info(self):
        """shutdown() should log info message."""
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig())

            with patch("fi.evals.framework.backends.kubernetes_backend.logger") as mock_logger:
                backend.shutdown()
                mock_logger.info.assert_called_once()


# ---------------------------------------------------------------------------
# TestKubernetesBackendStats
# ---------------------------------------------------------------------------

class TestKubernetesBackendStats:
    """Tests for KubernetesBackend.get_stats method."""

    def test_counts_by_status(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig(
                namespace="prod",
                image="custom:v1",
            ))

            h1 = TaskHandle(task_id="1", backend_name="kubernetes")
            h1._status = TaskStatus.PENDING
            h2 = TaskHandle(task_id="2", backend_name="kubernetes")
            h2._status = TaskStatus.RUNNING
            h3 = TaskHandle(task_id="3", backend_name="kubernetes")
            h3._status = TaskStatus.COMPLETED
            h4 = TaskHandle(task_id="4", backend_name="kubernetes")
            h4._status = TaskStatus.FAILED

            backend._handles = {"1": h1, "2": h2, "3": h3, "4": h4}

            stats = backend.get_stats()

            assert stats["namespace"] == "prod"
            assert stats["image"] == "custom:v1"
            assert stats["pending_tasks"] == 1
            assert stats["running_tasks"] == 1
            assert stats["completed_tasks"] == 1
            assert stats["failed_tasks"] == 1
            assert stats["total_tasks"] == 4

    def test_stats_empty(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig())
            stats = backend.get_stats()

            assert stats["pending_tasks"] == 0
            assert stats["running_tasks"] == 0
            assert stats["completed_tasks"] == 0
            assert stats["failed_tasks"] == 0
            assert stats["total_tasks"] == 0

    def test_stats_includes_config_info(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig(
                namespace="my-ns",
                image="eval:v2",
            ))
            stats = backend.get_stats()
            assert stats["namespace"] == "my-ns"
            assert stats["image"] == "eval:v2"


# ---------------------------------------------------------------------------
# TestKubernetesBackendJobLogs
# ---------------------------------------------------------------------------

class TestKubernetesBackendJobLogs:
    """Tests for KubernetesBackend.get_job_logs method."""

    def test_returns_logs(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig())

            handle = TaskHandle(task_id="t1", backend_name="kubernetes")
            backend._handles["t1"] = handle
            backend._job_names["t1"] = "eval-t1"

            # Mock pods list
            mock_pod = MagicMock()
            mock_pod.metadata.name = "eval-t1-abc"
            backend._core_api.list_namespaced_pod.return_value.items = [mock_pod]
            backend._core_api.read_namespaced_pod_log.return_value = "hello world\n"

            logs = backend.get_job_logs(handle)

            assert "eval-t1-abc" in logs
            assert logs["eval-t1-abc"] == "hello world\n"

    def test_unknown_task_raises(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig())
            handle = TaskHandle(task_id="unknown", backend_name="kubernetes")

            with pytest.raises(ValueError, match="Unknown task"):
                backend.get_job_logs(handle)

    def test_handles_read_failure(self):
        mock_k8s = _make_mock_k8s_modules()
        ApiException = mock_k8s["kubernetes.client.rest"].ApiException

        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig())

            handle = TaskHandle(task_id="t1", backend_name="kubernetes")
            backend._handles["t1"] = handle
            backend._job_names["t1"] = "eval-t1"

            mock_pod = MagicMock()
            mock_pod.metadata.name = "eval-t1-abc"
            backend._core_api.list_namespaced_pod.return_value.items = [mock_pod]
            backend._core_api.read_namespaced_pod_log.side_effect = ApiException("forbidden")

            logs = backend.get_job_logs(handle)

            assert "eval-t1-abc" in logs
            assert "<error reading logs:" in logs["eval-t1-abc"]


# ---------------------------------------------------------------------------
# TestKubernetesBackendBuildJobManifest
# ---------------------------------------------------------------------------

class TestKubernetesBackendBuildJobManifest:
    """Tests for _build_job_manifest method."""

    def test_manifest_structure(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig(
                labels={"team": "ml"},
                annotations={"note": "test"},
                image_pull_secrets=["my-secret"],
                service_account_name="eval-sa",
            ))

            job = backend._build_job_manifest("eval-abc123", "base64data")

            # Verify the client was called with the right structure
            mock_client = mock_k8s["kubernetes.client"]
            mock_client.V1Job.assert_called_once()
            mock_client.V1Container.assert_called_once()
            mock_client.V1PodSpec.assert_called_once()
            mock_client.V1ResourceRequirements.assert_called_once()

    def test_manifest_with_custom_image(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig(image="myimage:v1"))
            backend._build_job_manifest("eval-123", "payload")

            mock_client = mock_k8s["kubernetes.client"]
            # Verify V1Container was called with the correct image
            call_kwargs = mock_client.V1Container.call_args
            assert call_kwargs[1]["image"] == "myimage:v1"


# ---------------------------------------------------------------------------
# TestKubernetesBackendReadJobResult
# ---------------------------------------------------------------------------

class TestKubernetesBackendReadJobResult:
    """Tests for _read_job_result method."""

    def test_reads_success_result(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig())

            mock_pod = MagicMock()
            mock_pod.metadata.name = "eval-abc-pod"
            backend._core_api.list_namespaced_pod.return_value.items = [mock_pod]

            log_output = 'some debug output\n{"status": "success", "result": 42}\n'
            mock_resp = MagicMock()
            mock_resp.data = log_output.encode("utf-8")
            backend._core_api.read_namespaced_pod_log.return_value = mock_resp

            result = backend._read_job_result("eval-abc")
            assert result == 42

    def test_reads_error_result(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig())

            mock_pod = MagicMock()
            mock_pod.metadata.name = "eval-abc-pod"
            backend._core_api.list_namespaced_pod.return_value.items = [mock_pod]

            log_output = '{"status": "error", "error": "ZeroDivisionError"}\n'
            mock_resp = MagicMock()
            mock_resp.data = log_output.encode("utf-8")
            backend._core_api.read_namespaced_pod_log.return_value = mock_resp

            with pytest.raises(RuntimeError, match="ZeroDivisionError"):
                backend._read_job_result("eval-abc")

    def test_no_pods_found(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig())
            backend._core_api.list_namespaced_pod.return_value.items = []

            with pytest.raises(RuntimeError, match="No pods found"):
                backend._read_job_result("eval-xyz")

    def test_empty_logs(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig())

            mock_pod = MagicMock()
            mock_pod.metadata.name = "eval-abc-pod"
            backend._core_api.list_namespaced_pod.return_value.items = [mock_pod]
            mock_resp = MagicMock()
            mock_resp.data = b""
            backend._core_api.read_namespaced_pod_log.return_value = mock_resp

            with pytest.raises(RuntimeError, match="Empty container logs"):
                backend._read_job_result("eval-abc")

    def test_no_json_in_logs(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )
            backend = KubernetesBackend(KubernetesConfig())

            mock_pod = MagicMock()
            mock_pod.metadata.name = "eval-abc-pod"
            backend._core_api.list_namespaced_pod.return_value.items = [mock_pod]
            mock_resp = MagicMock()
            mock_resp.data = b"just plain text\nno json here\n"
            backend._core_api.read_namespaced_pod_log.return_value = mock_resp

            with pytest.raises(RuntimeError, match="No JSON result found"):
                backend._read_job_result("eval-abc")


# ---------------------------------------------------------------------------
# TestKubernetesBackendContextManager
# ---------------------------------------------------------------------------

class TestKubernetesBackendContextManager:
    """Tests for context manager support."""

    def test_context_manager(self):
        mock_k8s = _make_mock_k8s_modules()
        with _patch_kubernetes(mock_k8s):
            from fi.evals.framework.backends.kubernetes_backend import (
                KubernetesBackend,
                KubernetesConfig,
            )

            with KubernetesBackend(KubernetesConfig()) as backend:
                assert backend.name == "kubernetes"
            # Should not raise
