"""Tests for backend utilities module."""

import pytest
from unittest.mock import MagicMock, patch
import sys


class TestOptionalDependency:
    """Tests for OptionalDependency class."""

    def test_init(self):
        """Test initialization."""
        from fi.evals.framework.backends._utils import OptionalDependency

        dep = OptionalDependency("mymodule", "pip install mymodule", "myextra")
        assert dep.module_name == "mymodule"
        assert dep.install_hint == "pip install mymodule"
        assert dep.extra_name == "myextra"
        assert dep._module is None
        assert dep._checked is False
        assert dep._available is False

    def test_init_default_extra(self):
        """Test initialization with default extra name."""
        from fi.evals.framework.backends._utils import OptionalDependency

        dep = OptionalDependency("mymodule", "pip install mymodule")
        assert dep.extra_name == "mymodule"

    def test_is_available_installed(self):
        """Test is_available returns True when module is installed."""
        from fi.evals.framework.backends._utils import OptionalDependency

        # 'os' is always available
        dep = OptionalDependency("os", "pip install os")
        assert dep.is_available() is True
        assert dep._checked is True
        assert dep._available is True
        assert dep._module is not None

    def test_is_available_not_installed(self):
        """Test is_available returns False when module is not installed."""
        from fi.evals.framework.backends._utils import OptionalDependency

        dep = OptionalDependency("nonexistent_module_xyz", "pip install nonexistent")
        assert dep.is_available() is False
        assert dep._checked is True
        assert dep._available is False
        assert dep._module is None

    def test_is_available_caches_result(self):
        """Test that is_available caches the result."""
        from fi.evals.framework.backends._utils import OptionalDependency

        dep = OptionalDependency("os", "pip install os")

        # First call
        result1 = dep.is_available()
        assert dep._checked is True

        # Modify the internal state
        dep._available = False

        # Second call should return cached value
        result2 = dep.is_available()
        assert result2 is False  # Returns cached value

    def test_require_installed(self):
        """Test require returns module when installed."""
        from fi.evals.framework.backends._utils import OptionalDependency

        dep = OptionalDependency("os", "pip install os")
        module = dep.require()
        assert module is not None
        import os
        assert module is os

    def test_require_not_installed(self):
        """Test require raises ImportError when not installed."""
        from fi.evals.framework.backends._utils import OptionalDependency

        dep = OptionalDependency("nonexistent_module_xyz", "pip install nonexistent", "myextra")

        with pytest.raises(ImportError) as exc_info:
            dep.require()

        error_msg = str(exc_info.value)
        assert "nonexistent_module_xyz" in error_msg
        assert "pip install nonexistent" in error_msg
        assert "pip install fi-evals[myextra]" in error_msg

    def test_import_from_single(self):
        """Test import_from with single name."""
        from fi.evals.framework.backends._utils import OptionalDependency

        dep = OptionalDependency("os", "pip install os")
        path_module = dep.import_from("path")
        import os
        assert path_module is os.path

    def test_import_from_multiple(self):
        """Test import_from with multiple names."""
        from fi.evals.framework.backends._utils import OptionalDependency

        dep = OptionalDependency("os", "pip install os")
        path_module, sep = dep.import_from("path", "sep")
        import os
        assert path_module is os.path
        assert sep == os.sep

    def test_import_from_nested(self):
        """Test import_from with nested attribute."""
        from fi.evals.framework.backends._utils import OptionalDependency

        dep = OptionalDependency("os", "pip install os")
        join = dep.import_from("path.join")
        import os
        assert join is os.path.join

    def test_import_from_not_installed(self):
        """Test import_from raises ImportError when not installed."""
        from fi.evals.framework.backends._utils import OptionalDependency

        dep = OptionalDependency("nonexistent_module_xyz", "pip install nonexistent")

        with pytest.raises(ImportError):
            dep.import_from("something")


class TestPreConfiguredDependencies:
    """Tests for pre-configured optional dependencies."""

    def test_temporal_dependency(self):
        """Test TEMPORAL dependency is configured correctly."""
        from fi.evals.framework.backends._utils import TEMPORAL

        assert TEMPORAL.module_name == "temporalio"
        assert "pip install temporalio" in TEMPORAL.install_hint
        assert TEMPORAL.extra_name == "temporal"

    def test_celery_dependency(self):
        """Test CELERY dependency is configured correctly."""
        from fi.evals.framework.backends._utils import CELERY

        assert CELERY.module_name == "celery"
        assert "pip install" in CELERY.install_hint
        assert "celery" in CELERY.install_hint
        assert CELERY.extra_name == "celery"

    def test_ray_dependency(self):
        """Test RAY dependency is configured correctly."""
        from fi.evals.framework.backends._utils import RAY

        assert RAY.module_name == "ray"
        assert "pip install" in RAY.install_hint
        assert "ray" in RAY.install_hint
        assert RAY.extra_name == "ray"

    def test_kubernetes_dependency(self):
        """Test KUBERNETES dependency is configured correctly."""
        from fi.evals.framework.backends._utils import KUBERNETES

        assert KUBERNETES.module_name == "kubernetes"
        assert "pip install kubernetes" in KUBERNETES.install_hint
        assert KUBERNETES.extra_name == "kubernetes"


class TestCheckDependency:
    """Tests for check_dependency helper function."""

    def test_check_dependency_temporal(self):
        """Test check_dependency for temporal."""
        from fi.evals.framework.backends._utils import check_dependency, TEMPORAL

        # Reset state
        TEMPORAL._checked = False
        TEMPORAL._available = False

        # Result depends on whether temporalio is installed
        result = check_dependency("temporal")
        assert isinstance(result, bool)

    def test_check_dependency_celery(self):
        """Test check_dependency for celery."""
        from fi.evals.framework.backends._utils import check_dependency, CELERY

        # Reset state
        CELERY._checked = False
        CELERY._available = False

        result = check_dependency("celery")
        assert isinstance(result, bool)

    def test_check_dependency_ray(self):
        """Test check_dependency for ray."""
        from fi.evals.framework.backends._utils import check_dependency, RAY

        # Reset state
        RAY._checked = False
        RAY._available = False

        result = check_dependency("ray")
        assert isinstance(result, bool)

    def test_check_dependency_kubernetes(self):
        """Test check_dependency for kubernetes."""
        from fi.evals.framework.backends._utils import check_dependency, KUBERNETES

        # Reset state
        KUBERNETES._checked = False
        KUBERNETES._available = False

        result = check_dependency("kubernetes")
        assert isinstance(result, bool)

    def test_check_dependency_unknown(self):
        """Test check_dependency for unknown dependency."""
        from fi.evals.framework.backends._utils import check_dependency

        result = check_dependency("unknown_dep")
        assert result is False

    def test_check_dependency_case_insensitive(self):
        """Test check_dependency is case insensitive."""
        from fi.evals.framework.backends._utils import check_dependency

        # These should all work
        check_dependency("temporal")
        check_dependency("TEMPORAL")
        check_dependency("Temporal")


class TestLazyImports:
    """Tests for lazy imports in backends __init__.py."""

    def test_lazy_import_temporal_backend(self):
        """Test lazy import of TemporalBackend."""
        # This tests the __getattr__ mechanism
        with patch.dict("sys.modules", {"temporalio": MagicMock(), "temporalio.client": MagicMock()}):
            from fi.evals.framework.backends._utils import TEMPORAL
            TEMPORAL._checked = True
            TEMPORAL._available = True

            # Import through the package (triggers lazy import)
            from fi.evals.framework.backends import TemporalBackend
            assert TemporalBackend is not None

    def test_lazy_import_temporal_config(self):
        """Test lazy import of TemporalConfig."""
        with patch.dict("sys.modules", {"temporalio": MagicMock(), "temporalio.client": MagicMock()}):
            from fi.evals.framework.backends._utils import TEMPORAL
            TEMPORAL._checked = True
            TEMPORAL._available = True

            from fi.evals.framework.backends import TemporalConfig
            assert TemporalConfig is not None

    def test_lazy_import_celery_backend(self):
        """Test lazy import of CeleryBackend."""
        with patch.dict("sys.modules", {"celery": MagicMock(), "celery.result": MagicMock()}):
            from fi.evals.framework.backends._utils import CELERY
            CELERY._checked = True
            CELERY._available = True

            from fi.evals.framework.backends import CeleryBackend
            assert CeleryBackend is not None

    def test_lazy_import_ray_backend(self):
        """Test lazy import of RayBackend."""
        with patch.dict("sys.modules", {"ray": MagicMock(), "ray.exceptions": MagicMock()}):
            from fi.evals.framework.backends._utils import RAY
            RAY._checked = True
            RAY._available = True

            from fi.evals.framework.backends import RayBackend
            assert RayBackend is not None

    def test_lazy_import_unknown_raises(self):
        """Test that importing unknown attribute raises ImportError."""
        with pytest.raises(ImportError):
            from fi.evals.framework.backends import NonExistentBackend  # noqa: F401


class TestBackendInitExports:
    """Tests for backends __init__.py exports."""

    def test_base_exports_available(self):
        """Test that base classes are always available."""
        from fi.evals.framework.backends import (
            Backend,
            BackendConfig,
            TaskHandle,
            TaskStatus,
        )
        assert Backend is not None
        assert BackendConfig is not None
        assert TaskHandle is not None
        assert TaskStatus is not None

    def test_thread_pool_exports_available(self):
        """Test that ThreadPool classes are always available."""
        from fi.evals.framework.backends import (
            ThreadPoolBackend,
            ThreadPoolConfig,
        )
        assert ThreadPoolBackend is not None
        assert ThreadPoolConfig is not None

    def test_all_includes_optional_backends(self):
        """Test that __all__ includes optional backend names."""
        from fi.evals.framework import backends

        assert "TemporalBackend" in backends.__all__
        assert "TemporalConfig" in backends.__all__
        assert "CeleryBackend" in backends.__all__
        assert "CeleryConfig" in backends.__all__
        assert "RayBackend" in backends.__all__
        assert "RayConfig" in backends.__all__
