"""Jira Data Center integration routes.

This module delegates to the plugin-based router.  The
``jira_dc_integration_router`` name is kept for backward compatibility.
"""

from integrations.jira_dc.plugin import JiraDcPlugin

_plugin = JiraDcPlugin.from_env()
jira_dc_integration_router = _plugin.get_router()
