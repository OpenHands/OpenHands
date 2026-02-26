"""Slack integration routes — thin delegation layer.

This module preserves the original import path
``server.routes.integration.slack.slack_router`` for backwards compatibility
while delegating all route definitions to the plugin package at
``integrations.slack.routes``.

See issue #12978 for the migration plan.
"""

from integrations.slack.plugin import SlackPlugin

_plugin = SlackPlugin.from_env()
slack_router = _plugin.get_router()
