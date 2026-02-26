"""Slack integration plugin for OpenHands.

This package contains the self-contained Slack integration that can be
used as a standalone plugin. The primary entry point is :class:`SlackPlugin`.

Usage::

    from integrations.slack import SlackPlugin

    plugin = SlackPlugin.from_env()
    app.include_router(plugin.get_router())

See issue #12978 for the full migration plan.
"""

from integrations.slack.plugin import SlackPlugin

__all__ = ['SlackPlugin']
