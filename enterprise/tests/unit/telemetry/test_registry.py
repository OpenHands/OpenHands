"""Tests for the collector registry and decorator system."""

from typing import List
from unittest.mock import patch

import pytest

from enterprise.telemetry.base_collector import MetricResult, MetricsCollector
from enterprise.telemetry.registry import CollectorRegistry, register_collector


class TestCollectorRegistry:
    """Test cases for the CollectorRegistry class."""

    def setup_method(self):
        """Set up a fresh registry for each test."""
        self.registry = CollectorRegistry()

    def test_registry_initialization(self):
        """Test that registry initializes empty."""
        assert len(self.registry) == 0
        assert self.registry.list_collector_names() == []

    def test_register_collector_class(self):
        """Test registering a collector class."""

        class TestCollector(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'test_collector'

            def collect(self) -> List[MetricResult]:
                return [MetricResult(key='test', value=1)]

        self.registry.register(TestCollector)

        assert len(self.registry) == 1
        assert 'test_collector' in self.registry.list_collector_names()

    def test_register_invalid_collector(self):
        """Test registering a class that doesn't inherit from MetricsCollector."""

        class NotACollector:
            pass

        with pytest.raises(TypeError, match='must inherit from MetricsCollector'):
            self.registry.register(NotACollector)  # type: ignore[arg-type]

    def test_register_collector_with_instantiation_error(self):
        """Test registering a collector that fails to instantiate."""

        class FailingCollector(MetricsCollector):
            def __init__(self):
                raise ValueError('Cannot instantiate')

            @property
            def collector_name(self) -> str:
                return 'failing'

            def collect(self) -> List[MetricResult]:
                return []

        with pytest.raises(ValueError, match='Failed to instantiate collector'):
            self.registry.register(FailingCollector)

    def test_register_duplicate_collector_name(self):
        """Test registering collectors with duplicate names."""

        class Collector1(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'duplicate_name'

            def collect(self) -> List[MetricResult]:
                return []

        class Collector2(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'duplicate_name'

            def collect(self) -> List[MetricResult]:
                return []

        self.registry.register(Collector1)

        with pytest.raises(ValueError, match='already registered'):
            self.registry.register(Collector2)

    def test_register_same_collector_twice(self):
        """Test registering the same collector class twice (should be OK)."""

        class TestCollector(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'test'

            def collect(self) -> List[MetricResult]:
                return []

        self.registry.register(TestCollector)
        self.registry.register(TestCollector)  # Should not raise

        assert len(self.registry) == 1

    def test_get_all_collectors(self):
        """Test getting all registered collectors."""

        class Collector1(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'collector1'

            def collect(self) -> List[MetricResult]:
                return [MetricResult(key='metric1', value=1)]

        class Collector2(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'collector2'

            def collect(self) -> List[MetricResult]:
                return [MetricResult(key='metric2', value=2)]

        self.registry.register(Collector1)
        self.registry.register(Collector2)

        collectors = self.registry.get_all_collectors()
        assert len(collectors) == 2

        collector_names = [c.collector_name for c in collectors]
        assert 'collector1' in collector_names
        assert 'collector2' in collector_names

    def test_get_all_collectors_with_instantiation_failure(self):
        """Test get_all_collectors when one collector fails to instantiate."""

        class GoodCollector(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'good'

            def collect(self) -> List[MetricResult]:
                return []

        class BadCollector(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'bad'

            def __init__(self):
                raise RuntimeError('Instantiation failed')

            def collect(self) -> List[MetricResult]:
                return []

        # Register the good collector first
        self.registry.register(GoodCollector)

        # Manually add the bad collector to simulate registration
        self.registry._collectors['bad'] = BadCollector

        # Should return only the good collector, log error for bad one
        with patch('enterprise.telemetry.registry.logger') as mock_logger:
            collectors = self.registry.get_all_collectors()

            assert len(collectors) == 1
            assert collectors[0].collector_name == 'good'
            mock_logger.error.assert_called_once()

    def test_get_collector_by_name(self):
        """Test getting a specific collector by name."""

        class TestCollector(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'test_collector'

            def collect(self) -> List[MetricResult]:
                return [MetricResult(key='test', value=42)]

        self.registry.register(TestCollector)

        collector = self.registry.get_collector_by_name('test_collector')
        assert collector.collector_name == 'test_collector'

        results = collector.collect()
        assert len(results) == 1
        assert results[0].key == 'test'
        assert results[0].value == 42

    def test_get_collector_by_nonexistent_name(self):
        """Test getting a collector that doesn't exist."""

        with pytest.raises(KeyError, match='No collector registered with name'):
            self.registry.get_collector_by_name('nonexistent')

    def test_unregister_collector(self):
        """Test unregistering a collector."""

        class TestCollector(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'test'

            def collect(self) -> List[MetricResult]:
                return []

        self.registry.register(TestCollector)
        assert len(self.registry) == 1

        result = self.registry.unregister('test')
        assert result is True
        assert len(self.registry) == 0
        assert 'test' not in self.registry.list_collector_names()

    def test_unregister_nonexistent_collector(self):
        """Test unregistering a collector that doesn't exist."""

        result = self.registry.unregister('nonexistent')
        assert result is False

    def test_clear_registry(self):
        """Test clearing all collectors from registry."""

        class Collector1(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'collector1'

            def collect(self) -> List[MetricResult]:
                return []

        class Collector2(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'collector2'

            def collect(self) -> List[MetricResult]:
                return []

        self.registry.register(Collector1)
        self.registry.register(Collector2)
        assert len(self.registry) == 2

        self.registry.clear()
        assert len(self.registry) == 0
        assert self.registry.list_collector_names() == []

    def test_discover_collectors_invalid_package(self):
        """Test discovering collectors in a non-existent package."""

        with pytest.raises(ImportError):
            self.registry.discover_collectors('nonexistent.package')

    def test_registry_repr(self):
        """Test string representation of registry."""

        repr_str = repr(self.registry)
        assert 'CollectorRegistry' in repr_str
        assert 'collectors=0' in repr_str


class TestRegisterCollectorDecorator:
    """Test cases for the @register_collector decorator."""

    def setup_method(self):
        """Set up for each test."""
        # Clear the global registry
        from enterprise.telemetry.registry import collector_registry

        collector_registry.clear()

    def test_register_collector_decorator(self):
        """Test the @register_collector decorator."""

        @register_collector('decorated_collector')
        class DecoratedCollector(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'decorated_collector'

            def collect(self) -> List[MetricResult]:
                return [MetricResult(key='decorated', value=True)]

        from enterprise.telemetry.registry import collector_registry

        assert 'decorated_collector' in collector_registry.list_collector_names()

        collector = collector_registry.get_collector_by_name('decorated_collector')
        assert collector.collector_name == 'decorated_collector'

        results = collector.collect()
        assert len(results) == 1
        assert results[0].key == 'decorated'
        assert results[0].value is True

    def test_decorator_with_registration_failure(self):
        """Test decorator when registration fails."""

        with patch(
            'enterprise.telemetry.registry.collector_registry.register'
        ) as mock_register:
            mock_register.side_effect = ValueError('Registration failed')

            with patch('enterprise.telemetry.registry.logger') as mock_logger:

                @register_collector('failing_collector')
                class FailingCollector(MetricsCollector):
                    @property
                    def collector_name(self) -> str:
                        return 'failing_collector'

                    def collect(self) -> List[MetricResult]:
                        return []

                # Should not raise exception, but should log error
                mock_logger.error.assert_called_once()

                # Class should still be returned unchanged
                assert FailingCollector is not None

    def test_decorator_returns_original_class(self):
        """Test that decorator returns the original class unchanged."""

        @register_collector('test_class')
        class TestClass(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'test_class'

            def collect(self) -> List[MetricResult]:
                return []

            def custom_method(self):
                return 'custom'

        # Class should be unchanged
        assert hasattr(TestClass, 'custom_method')
        instance = TestClass()
        assert instance.custom_method() == 'custom'
