"""Linear integration routes.

This module delegates to the plugin-based router.  The
``linear_integration_router`` name is kept for backward compatibility.
"""

from integrations.linear.plugin import LinearPlugin

_plugin = LinearPlugin.from_env()
linear_integration_router = _plugin.get_router()
