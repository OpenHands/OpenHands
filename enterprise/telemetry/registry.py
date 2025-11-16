"""Collector registry for automatic discovery and management of metrics collectors.

This module provides the registry system that allows collectors to be automatically
discovered and executed by the collection system using the @register_collector decorator.
"""

import importlib
import logging
import pkgutil
from typing import Dict, List, Type

from .base_collector import MetricsCollector

logger = logging.getLogger(__name__)


class CollectorRegistry:
    """Registry for metrics collectors.

    This class maintains a registry of all metrics collectors that have been
    registered using the @register_collector decorator. It provides methods
    to retrieve collectors and discover them automatically.
    """

    def __init__(self):
        """Initialize an empty collector registry."""
        self._collectors: Dict[str, Type[MetricsCollector]] = {}

    def register(self, collector_class: Type[MetricsCollector]) -> None:
        """Register a collector class.

        Args:
            collector_class: The collector class to register

        Raises:
            ValueError: If the collector name is already registered
            TypeError: If the collector class doesn't inherit from MetricsCollector
        """
        if not issubclass(collector_class, MetricsCollector):
            raise TypeError(
                f'Collector class {collector_class.__name__} must inherit from MetricsCollector'
            )

        # Create a temporary instance to get the collector name
        try:
            collector_instance = collector_class()
            collector_name = collector_instance.collector_name
        except Exception as e:
            raise ValueError(
                f'Failed to instantiate collector {collector_class.__name__}: {e}'
            ) from e

        if collector_name in self._collectors:
            existing_class = self._collectors[collector_name]
            if existing_class != collector_class:
                raise ValueError(
                    f"Collector name '{collector_name}' is already registered "
                    f'by {existing_class.__name__}'
                )
            # Same class being registered again - this is OK (e.g., during testing)
            logger.debug(f"Collector '{collector_name}' already registered, skipping")
            return

        self._collectors[collector_name] = collector_class
        logger.info(f'Registered collector: {collector_name}')

    def get_all_collectors(self) -> List[MetricsCollector]:
        """Get instances of all registered collectors.

        Returns:
            List of instantiated collector objects

        Raises:
            Exception: If any collector fails to instantiate, it will be logged
                      and excluded from the returned list
        """
        collectors = []
        for name, collector_class in self._collectors.items():
            try:
                collector = collector_class()
                collectors.append(collector)
            except Exception as e:
                logger.error(f"Failed to instantiate collector '{name}': {e}")
                # Continue with other collectors rather than failing completely

        return collectors

    def get_collector_by_name(self, name: str) -> MetricsCollector:
        """Get a specific collector by name.

        Args:
            name: The collector name to retrieve

        Returns:
            An instance of the requested collector

        Raises:
            KeyError: If no collector with the given name is registered
            Exception: If the collector fails to instantiate
        """
        if name not in self._collectors:
            raise KeyError(f"No collector registered with name '{name}'")

        collector_class = self._collectors[name]
        return collector_class()

    def list_collector_names(self) -> List[str]:
        """Get a list of all registered collector names.

        Returns:
            List of collector names
        """
        return list(self._collectors.keys())

    def unregister(self, name: str) -> bool:
        """Unregister a collector by name.

        This is primarily useful for testing scenarios.

        Args:
            name: The collector name to unregister

        Returns:
            True if the collector was unregistered, False if it wasn't found
        """
        if name in self._collectors:
            del self._collectors[name]
            logger.info(f'Unregistered collector: {name}')
            return True
        return False

    def clear(self) -> None:
        """Clear all registered collectors.

        This is primarily useful for testing scenarios.
        """
        count = len(self._collectors)
        self._collectors.clear()
        logger.info(f'Cleared {count} registered collectors')

    def discover_collectors(self, package_path: str) -> int:
        """Auto-discover collectors in a package.

        This method will import all modules in the specified package path,
        which will trigger the @register_collector decorators to register
        their collectors.

        Args:
            package_path: Python package path to scan (e.g., 'enterprise.telemetry.collectors')

        Returns:
            Number of new collectors discovered and registered

        Raises:
            ImportError: If the package cannot be imported
        """
        initial_count = len(self._collectors)

        try:
            package = importlib.import_module(package_path)
        except ImportError as e:
            logger.error(f"Failed to import package '{package_path}': {e}")
            raise

        # Import all submodules in the package
        if hasattr(package, '__path__'):
            for _, module_name, _ in pkgutil.iter_modules(package.__path__):
                full_module_name = f'{package_path}.{module_name}'
                try:
                    importlib.import_module(full_module_name)
                    logger.debug(f'Imported module: {full_module_name}')
                except Exception as e:
                    logger.error(f"Failed to import module '{full_module_name}': {e}")

        new_count = len(self._collectors) - initial_count
        logger.info(
            f"Discovered {new_count} new collectors in package '{package_path}'"
        )
        return new_count

    def __len__(self) -> int:
        """Return the number of registered collectors."""
        return len(self._collectors)

    def __repr__(self) -> str:
        """String representation of the registry."""
        return f'<CollectorRegistry(collectors={len(self._collectors)})>'


# Global registry instance
collector_registry = CollectorRegistry()


def register_collector(name: str):
    """Decorator to register a collector.

    This decorator automatically registers a collector class with the global
    collector registry when the module is imported.

    Args:
        name: The name to register the collector under (optional, will use
              collector_name property if not provided)

    Returns:
        The decorator function

    Example:
        @register_collector("system_metrics")
        class SystemMetricsCollector(MetricsCollector):
            @property
            def collector_name(self) -> str:
                return "system_metrics"

            def collect(self) -> List[MetricResult]:
                return [MetricResult("cpu_usage", 75.5)]
    """

    def decorator(cls: Type[MetricsCollector]) -> Type[MetricsCollector]:
        """The actual decorator function.

        Args:
            cls: The collector class to register

        Returns:
            The original class (unchanged)
        """
        try:
            collector_registry.register(cls)
        except Exception as e:
            logger.error(f'Failed to register collector {cls.__name__}: {e}')
            # Don't raise the exception to avoid breaking module imports

        return cls

    return decorator
