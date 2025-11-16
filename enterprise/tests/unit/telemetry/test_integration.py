"""Integration tests for the telemetry collection framework.

These tests verify that the entire collection system works together,
including automatic discovery, registration, and execution of collectors.
"""

from typing import List
from unittest.mock import MagicMock, patch

from telemetry.base_collector import MetricResult, MetricsCollector
from telemetry.registry import CollectorRegistry, register_collector


class TestTelemetryFrameworkIntegration:
    """Integration tests for the complete telemetry framework."""

    def setup_method(self):
        """Set up for each test."""
        self.registry = CollectorRegistry()

    def test_end_to_end_collection_flow(self):
        """Test the complete flow from registration to collection."""

        # Define test collectors using the decorator
        @register_collector('integration_test_collector1')
        class TestCollector1(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'integration_test_collector1'

            def collect(self) -> List[MetricResult]:
                return [
                    MetricResult(key='metric1', value=100),
                    MetricResult(key='metric2', value='test_value'),
                ]

        @register_collector('integration_test_collector2')
        class TestCollector2(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'integration_test_collector2'

            def collect(self) -> List[MetricResult]:
                return [
                    MetricResult(key='metric3', value=200.5),
                    MetricResult(key='metric4', value=True),
                ]

        # Register collectors with our test registry
        self.registry.register(TestCollector1)
        self.registry.register(TestCollector2)

        # Verify registration
        assert len(self.registry) == 2
        collector_names = self.registry.list_collector_names()
        assert 'integration_test_collector1' in collector_names
        assert 'integration_test_collector2' in collector_names

        # Collect all metrics
        all_collectors = self.registry.get_all_collectors()
        all_results = []

        for collector in all_collectors:
            results = collector.collect()
            all_results.extend(results)

        # Verify we got all expected metrics
        assert len(all_results) == 4

        result_dict = {r.key: r.value for r in all_results}
        assert result_dict['metric1'] == 100
        assert result_dict['metric2'] == 'test_value'
        assert result_dict['metric3'] == 200.5
        assert result_dict['metric4'] is True

    def test_collector_discovery_simulation(self):
        """Test simulated collector discovery process."""

        # Create collectors that would be discovered
        class DiscoveredCollector1(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'discovered1'

            def collect(self) -> List[MetricResult]:
                return [MetricResult(key='discovered_metric1', value=42)]

        class DiscoveredCollector2(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'discovered2'

            def collect(self) -> List[MetricResult]:
                return [MetricResult(key='discovered_metric2', value='discovered')]

        # Simulate the discovery process
        discovered_collectors = [DiscoveredCollector1, DiscoveredCollector2]

        for collector_class in discovered_collectors:
            self.registry.register(collector_class)

        # Verify discovery worked
        assert len(self.registry) == 2

        # Test collection from discovered collectors
        collectors = self.registry.get_all_collectors()
        all_metrics = []

        for collector in collectors:
            metrics = collector.collect()
            all_metrics.extend(metrics)

        assert len(all_metrics) == 2
        metric_keys = [m.key for m in all_metrics]
        assert 'discovered_metric1' in metric_keys
        assert 'discovered_metric2' in metric_keys

    def test_mixed_collector_success_and_failure(self):
        """Test collection when some collectors succeed and others fail."""

        class SuccessfulCollector(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'successful'

            def collect(self) -> List[MetricResult]:
                return [MetricResult(key='success_metric', value=1)]

        class FailingCollector(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return 'failing'

            def collect(self) -> List[MetricResult]:
                raise RuntimeError('Collection failed')

        self.registry.register(SuccessfulCollector)
        self.registry.register(FailingCollector)

        collectors = self.registry.get_all_collectors()
        successful_results = []
        failed_collectors = []

        for collector in collectors:
            try:
                results = collector.collect()
                successful_results.extend(results)
            except Exception as e:
                failed_collectors.append((collector.collector_name, str(e)))

        # Verify we got results from successful collector
        assert len(successful_results) == 1
        assert successful_results[0].key == 'success_metric'

        # Verify we tracked the failure
        assert len(failed_collectors) == 1
        assert failed_collectors[0][0] == 'failing'
        assert 'Collection failed' in failed_collectors[0][1]

    def test_real_collector_integration(self):
        """Test integration with actual collector implementations."""
        from telemetry.collectors.health_check import HealthCheckCollector

        # Mock dependencies using context managers
        with patch(
            'telemetry.collectors.health_check.platform'
        ) as mock_platform, patch(
            'telemetry.collectors.health_check.session_maker'
        ) as mock_session_maker:
            # Mock dependencies
            mock_platform.system.return_value = 'Linux'
            mock_platform.release.return_value = '5.4.0'
            mock_platform.python_version.return_value = '3.11.0'

            mock_session = MagicMock()
            mock_session_maker.return_value.__enter__.return_value = mock_session

            # Register real collector
            self.registry.register(HealthCheckCollector)

            # Collect metrics
            collectors = self.registry.get_all_collectors()
            assert len(collectors) == 1

            collector = collectors[0]
            assert collector.collector_name == 'health_check'

            results = collector.collect()
            assert len(results) > 0

            # Verify we got expected health check metrics
            result_keys = [r.key for r in results]
            assert 'platform_system' in result_keys
            assert 'database_healthy' in result_keys

    def test_collector_isolation(self):
        """Test that collectors are properly isolated from each other."""

        class StatefulCollector(MetricsCollector):
            def __init__(self):
                self.call_count = 0

            @property
            def collector_name(self) -> str:
                return 'stateful'

            def collect(self) -> List[MetricResult]:
                self.call_count += 1
                return [MetricResult(key='call_count', value=self.call_count)]

        self.registry.register(StatefulCollector)

        # Get multiple instances and verify they're independent
        collector1 = self.registry.get_collector_by_name('stateful')
        collector2 = self.registry.get_collector_by_name('stateful')

        # They should be different instances
        assert collector1 is not collector2

        # Each should have independent state
        results1 = collector1.collect()
        results2 = collector2.collect()

        assert results1[0].value == 1
        assert results2[0].value == 1  # Fresh instance, starts at 1

    def test_large_scale_collection(self):
        """Test collection with many collectors to verify scalability."""

        # Create many collectors
        num_collectors = 50

        for i in range(num_collectors):

            class ScaleTestCollector(MetricsCollector):
                def __init__(self, collector_id=i):
                    self.collector_id = collector_id

                @property
                def collector_name(self) -> str:
                    return f'scale_test_{self.collector_id}'

                def collect(self) -> List[MetricResult]:
                    return [
                        MetricResult(
                            key=f'metric_{self.collector_id}', value=self.collector_id
                        ),
                        MetricResult(
                            key=f'squared_{self.collector_id}',
                            value=self.collector_id**2,
                        ),
                    ]

            # Create a unique class for each collector to avoid registration conflicts
            collector_class = type(
                f'ScaleTestCollector{i}',
                (MetricsCollector,),
                {
                    '__init__': lambda self, cid=i: setattr(self, 'collector_id', cid),
                    'collector_name': property(
                        lambda self: f'scale_test_{self.collector_id}'
                    ),
                    'collect': lambda self: [
                        MetricResult(
                            key=f'metric_{self.collector_id}', value=self.collector_id
                        ),
                        MetricResult(
                            key=f'squared_{self.collector_id}',
                            value=self.collector_id**2,
                        ),
                    ],
                },
            )

            self.registry.register(collector_class)

        # Verify all collectors were registered
        assert len(self.registry) == num_collectors

        # Collect all metrics
        all_collectors = self.registry.get_all_collectors()
        all_results = []

        for collector in all_collectors:
            results = collector.collect()
            all_results.extend(results)

        # Verify we got all expected metrics
        assert len(all_results) == num_collectors * 2  # 2 metrics per collector

        # Verify metric values are correct
        metric_values = {}
        for result in all_results:
            metric_values[result.key] = result.value

        for i in range(num_collectors):
            assert metric_values[f'metric_{i}'] == i
            assert metric_values[f'squared_{i}'] == i**2

    def test_registry_thread_safety_simulation(self):
        """Test registry behavior under simulated concurrent access."""

        import threading
        import time

        results = []
        errors = []

        def register_and_collect(collector_id):
            try:

                class ThreadCollector(MetricsCollector):
                    @property
                    def collector_name(self) -> str:
                        return f'thread_collector_{collector_id}'

                    def collect(self) -> List[MetricResult]:
                        time.sleep(0.001)  # Simulate work
                        return [
                            MetricResult(
                                key=f'thread_metric_{collector_id}', value=collector_id
                            )
                        ]

                # Create unique class to avoid conflicts
                collector_class = type(
                    f'ThreadCollector{collector_id}',
                    (MetricsCollector,),
                    {
                        'collector_name': property(
                            lambda self: f'thread_collector_{collector_id}'
                        ),
                        'collect': lambda self: [
                            MetricResult(
                                key=f'thread_metric_{collector_id}', value=collector_id
                            )
                        ],
                    },
                )

                self.registry.register(collector_class)

                collector = self.registry.get_collector_by_name(
                    f'thread_collector_{collector_id}'
                )
                thread_results = collector.collect()
                results.extend(thread_results)

            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=register_and_collect, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f'Errors occurred: {errors}'
        assert len(results) == 10
        assert len(self.registry) == 10
