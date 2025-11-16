"""Base collector interface for the OpenHands Enterprise Telemetry Framework.

This module defines the abstract base class that all metrics collectors must inherit from,
providing a consistent interface for the collection system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List


@dataclass
class MetricResult:
    """Represents a single metric result from a collector.

    Attributes:
        key: The metric name/identifier
        value: The metric value (can be any JSON-serializable type)
    """

    key: str
    value: Any

    def __post_init__(self):
        """Validate the metric result after initialization."""
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError('Metric key must be a non-empty string')


class MetricsCollector(ABC):
    """Abstract base class for metrics collectors.

    All metrics collectors must inherit from this class and implement the required
    abstract methods. This ensures a consistent interface for the collection system.
    """

    @abstractmethod
    def collect(self) -> List[MetricResult]:
        """Collect metrics and return results.

        This method should perform the actual metrics collection logic and return
        a list of MetricResult objects representing the collected metrics.

        Returns:
            List of MetricResult objects containing the collected metrics

        Raises:
            Exception: If collection fails, the exception will be caught and logged
                      by the collection system
        """
        pass

    @property
    @abstractmethod
    def collector_name(self) -> str:
        """Unique name for this collector.

        This name is used for identification in logs and registry management.
        It should be unique across all collectors in the system.

        Returns:
            A unique string identifier for this collector
        """
        pass

    def should_collect(self) -> bool:
        """Determine if this collector should run during the current collection cycle.

        Override this method to add collection conditions (e.g., time-based collection,
        conditional collection based on system state, etc.).

        Returns:
            True if the collector should run, False otherwise
        """
        return True

    def __repr__(self) -> str:
        """String representation of the collector."""
        return f"<{self.__class__.__name__}(name='{self.collector_name}')>"
