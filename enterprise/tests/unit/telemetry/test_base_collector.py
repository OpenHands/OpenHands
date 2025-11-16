"""Tests for the base collector interface."""

from abc import ABC
from typing import List

import pytest

from enterprise.telemetry.base_collector import MetricResult, MetricsCollector


class TestMetricResult:
    """Test cases for the MetricResult dataclass."""

    def test_metric_result_creation(self):
        """Test creating a MetricResult with basic values."""
        result = MetricResult(key='test_metric', value=42)
        assert result.key == 'test_metric'
        assert result.value == 42

    def test_metric_result_with_string_value(self):
        """Test creating a MetricResult with string value."""
        result = MetricResult(key='status', value='healthy')
        assert result.key == 'status'
        assert result.value == 'healthy'

    def test_metric_result_with_float_value(self):
        """Test creating a MetricResult with float value."""
        result = MetricResult(key='cpu_usage', value=75.5)
        assert result.key == 'cpu_usage'
        assert result.value == 75.5

    def test_metric_result_equality(self):
        """Test MetricResult equality comparison."""
        result1 = MetricResult(key='test', value=100)
        result2 = MetricResult(key='test', value=100)
        result3 = MetricResult(key='test', value=200)

        assert result1 == result2
        assert result1 != result3

    def test_metric_result_repr(self):
        """Test MetricResult string representation."""
        result = MetricResult(key='test_metric', value=42)
        repr_str = repr(result)
        assert 'test_metric' in repr_str
        assert '42' in repr_str


class TestMetricsCollector:
    """Test cases for the MetricsCollector abstract base class."""

    def test_metrics_collector_is_abstract(self):
        """Test that MetricsCollector cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MetricsCollector()  # type: ignore[abstract]

    def test_metrics_collector_inheritance(self):
        """Test that MetricsCollector is properly abstract."""
        assert issubclass(MetricsCollector, ABC)

        # Check that the required methods are abstract
        abstract_methods = MetricsCollector.__abstractmethods__
        assert 'collect' in abstract_methods
        assert 'collector_name' in abstract_methods

    def test_concrete_collector_implementation(self):
        """Test that a concrete collector can be implemented."""

        class TestCollector(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'test_collector'

            def collect(self) -> List[MetricResult]:
                return [
                    MetricResult(key='metric1', value=10),
                    MetricResult(key='metric2', value='test'),
                ]

        collector = TestCollector()
        assert collector.collector_name == 'test_collector'

        results = collector.collect()
        assert len(results) == 2
        assert results[0].key == 'metric1'
        assert results[0].value == 10
        assert results[1].key == 'metric2'
        assert results[1].value == 'test'

    def test_collector_with_empty_results(self):
        """Test collector that returns empty results."""

        class EmptyCollector(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'empty_collector'

            def collect(self) -> List[MetricResult]:
                return []

        collector = EmptyCollector()
        results = collector.collect()
        assert results == []

    def test_collector_with_exception(self):
        """Test collector that raises an exception."""

        class FailingCollector(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'failing_collector'

            def collect(self) -> List[MetricResult]:
                raise RuntimeError('Collection failed')

        collector = FailingCollector()
        with pytest.raises(RuntimeError, match='Collection failed'):
            collector.collect()

    def test_collector_name_property(self):
        """Test that collector_name is properly implemented as a property."""

        class NamedCollector(MetricsCollector):
            def __init__(self, name: str):
                self._name = name

            @property
            def collector_name(self) -> str:
                return self._name

            def collect(self) -> List[MetricResult]:
                return []

        collector = NamedCollector('dynamic_name')
        assert collector.collector_name == 'dynamic_name'

    def test_incomplete_collector_implementation(self):
        """Test that incomplete implementations cannot be instantiated."""

        # Missing collect method
        class IncompleteCollector1(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'incomplete'

        with pytest.raises(TypeError):
            IncompleteCollector1()  # type: ignore[abstract]

        # Missing collector_name property
        class IncompleteCollector2(MetricsCollector):
            def collect(self) -> List[MetricResult]:
                return []

        with pytest.raises(TypeError):
            IncompleteCollector2()  # type: ignore[abstract]
