"""Slack integration plugin for OpenHands.

This module provides the plugin entry point for the Slack integration,
encapsulating all configuration, routes, and lifecycle management
in a single class. This is the first step toward a fully standalone
plugin architecture (see #12978, #12971, #12972).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import APIRouter

_logger = logging.getLogger(__name__)


@dataclass
class SlackPluginConfig:
    """Configuration for the Slack plugin.

    All values are read from environment variables at startup and
    injected into the plugin rather than imported from server constants.
    """

    client_id: str | None = None
    client_secret: str | None = None
    signing_secret: str | None = None
    webhooks_enabled: bool = False
    keycloak_client_id: str = ''
    keycloak_realm_name: str = ''
    keycloak_server_url_ext: str = ''
    host_url: str = ''
    jwt_secret: str | None = None


class SlackPlugin:
    """Slack integration plugin for OpenHands.

    Encapsulates the entire Slack integration: OAuth flow, webhook
    event handling, form interactions, and callback processing.

    Usage::

        config = SlackPluginConfig.from_env()
        plugin = SlackPlugin(config)
        plugin.initialize()
        app.include_router(plugin.get_router())
    """

    def __init__(self, config: SlackPluginConfig) -> None:
        self._config = config
        self._router: APIRouter | None = None
        self._initialized = False

    @property
    def config(self) -> SlackPluginConfig:
        return self._config

    @property
    def name(self) -> str:
        return 'slack'

    def initialize(self) -> None:
        """Perform any one-time setup (connections, background tasks, etc.)."""
        if self._initialized:
            return
        _logger.info('[SlackPlugin] Initializing Slack integration plugin')
        self._initialized = True

    def shutdown(self) -> None:
        """Clean up resources on application shutdown."""
        _logger.info('[SlackPlugin] Shutting down Slack integration plugin')
        self._initialized = False

    def get_router(self) -> APIRouter:
        """Return the FastAPI router for this plugin.

        Lazily imports routes to avoid circular dependency issues.
        """
        if self._router is None:
            from integrations.slack.routes import create_slack_router

            self._router = create_slack_router(self._config)
        return self._router

    @staticmethod
    def from_env() -> 'SlackPlugin':
        """Create a SlackPlugin instance configured from environment variables."""
        import os

        from integrations.utils import HOST_URL
        from server.auth.constants import (
            KEYCLOAK_CLIENT_ID,
            KEYCLOAK_REALM_NAME,
            KEYCLOAK_SERVER_URL_EXT,
        )

        config = SlackPluginConfig(
            client_id=os.environ.get('SLACK_CLIENT_ID'),
            client_secret=os.environ.get('SLACK_CLIENT_SECRET'),
            signing_secret=os.environ.get('SLACK_SIGNING_SECRET'),
            webhooks_enabled=os.environ.get('SLACK_WEBHOOKS_ENABLED', '0')
            in ('1', 'true'),
            keycloak_client_id=KEYCLOAK_CLIENT_ID,
            keycloak_realm_name=KEYCLOAK_REALM_NAME,
            keycloak_server_url_ext=KEYCLOAK_SERVER_URL_EXT,
            host_url=HOST_URL,
        )
        plugin = SlackPlugin(config)
        plugin.initialize()
        return plugin
