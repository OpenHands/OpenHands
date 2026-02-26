import os
from dataclasses import dataclass

from fastapi import APIRouter


@dataclass
class JiraDcPluginConfig:
    """Configuration for the Jira Data Center plugin."""

    webhooks_enabled: bool = False
    base_url: str = ''
    client_id: str = ''
    client_secret: str = ''
    enable_oauth: bool = True
    web_host: str = ''


class JiraDcPlugin:
    """Plugin entry point for Jira Data Center integration.

    Holds all configuration and provides lifecycle hooks plus
    a factory for the FastAPI router.
    """

    def __init__(self, config: JiraDcPluginConfig) -> None:
        self._config = config
        self._router: APIRouter | None = None
        self._initialized = False

    @property
    def config(self) -> JiraDcPluginConfig:
        return self._config

    def get_router(self) -> APIRouter:
        if self._router is None:
            from integrations.jira_dc.routes import create_jira_dc_router

            self._router = create_jira_dc_router(self._config)
        return self._router

    def initialize(self) -> None:
        self._initialized = True

    def shutdown(self) -> None:
        self._initialized = False

    @staticmethod
    def from_env() -> 'JiraDcPlugin':
        from server.auth.constants import (
            JIRA_DC_BASE_URL,
            JIRA_DC_CLIENT_ID,
            JIRA_DC_CLIENT_SECRET,
            JIRA_DC_ENABLE_OAUTH,
        )
        from server.constants import WEB_HOST

        config = JiraDcPluginConfig(
            webhooks_enabled=os.environ.get('JIRA_DC_WEBHOOKS_ENABLED', '0')
            in ('1', 'true'),
            base_url=JIRA_DC_BASE_URL,
            client_id=JIRA_DC_CLIENT_ID,
            client_secret=JIRA_DC_CLIENT_SECRET,
            enable_oauth=JIRA_DC_ENABLE_OAUTH,
            web_host=WEB_HOST,
        )
        return JiraDcPlugin(config)
