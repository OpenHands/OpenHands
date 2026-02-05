"""Automatic tracing proxy for OpenHands runtime plugins."""

from typing import Any

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import Action
from openhands.events.observation import Observation
from openhands.runtime.plugins.requirement import Plugin

try:
    from lmnr import Laminar, observe as laminar_observe

    LAMINAR_AVAILABLE = True
except ImportError:
    LAMINAR_AVAILABLE = False
    logger.debug('Laminar not available, tracing disabled')


class TracingPluginProxy(Plugin):
    """Proxy that adds distributed tracing to any plugin.

    Automatically wraps plugin.initialize() and plugin.run() with Laminar spans.
    All other attributes are delegated to the wrapped plugin.
    """

    def __init__(self, wrapped_plugin: Plugin):
        """Wrap a plugin with tracing.

        Args:
            wrapped_plugin: The plugin instance to wrap
        """
        self._plugin = wrapped_plugin
        self.name = wrapped_plugin.name
        logger.debug(f'Created tracing proxy for plugin: {self.name}')

    async def initialize(self, *args, **kwargs) -> None:
        """Initialize plugin with tracing."""
        if not LAMINAR_AVAILABLE or not Laminar.is_initialized():
            return await self._plugin.initialize(*args, **kwargs)

        @laminar_observe(
            name=f'plugin.{self.name}.initialize',
            span_type='DEFAULT',
            metadata={'plugin': self.name, 'component': 'runtime'},
        )
        async def _traced_init():
            return await self._plugin.initialize(*args, **kwargs)

        return await _traced_init()

    async def run(self, action: Action) -> Observation:
        """Run plugin with tracing."""
        if not LAMINAR_AVAILABLE or not Laminar.is_initialized():
            return await self._plugin.run(action)

        @laminar_observe(
            name=f'plugin.{self.name}.run',
            span_type='TOOL',
            metadata={
                'plugin': self.name,
                'action_type': action.__class__.__name__,
                'component': 'runtime',
            },
        )
        async def _traced_run():
            return await self._plugin.run(action)

        return await _traced_run()

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the wrapped plugin."""
        return getattr(self._plugin, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attributes on wrapped plugin (except internal proxy state)."""
        if name in ('_plugin', 'name'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._plugin, name, value)
