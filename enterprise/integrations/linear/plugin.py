import os
from dataclasses import dataclass

from fastapi import APIRouter


@dataclass
class LinearPluginConfig:
    """Configuration for the Linear plugin."""

    webhooks_enabled: bool = False
    client_id: str = ""
    client_secret: str = ""
    web_host: str = ""


class LinearPlugin:
    """Plugin entry point for Linear integration.

    Holds all configuration and provides lifecycle hooks plus
    a factory for the FastAPI router.
    """

    def __init__(self, config: LinearPluginConfig) -> None:
        self._config = config
        self._router: APIRouter | None = None
        self._initialized: bool = False

    @property
    def config(self) -> LinearPluginConfig:
        return self._config

    def get_router(self) -> APIRouter:
        if self._router is None:
            from integrations.linear.routes import create_linear_router

            self._router = create_linear_router(self._config)
        return self._router

    def initialize(self) -> None:
        self._initialized = True

    def shutdown(self) -> None:
        self._initialized = False

    @staticmethod
    def from_env() -> "LinearPlugin":
        from server.auth.constants import LINEAR_CLIENT_ID, LINEAR_CLIENT_SECRET
        from server.constants import WEB_HOST

        config = LinearPluginConfig(
            webhooks_enabled=os.environ.get("LINEAR_WEBHOOKS_ENABLED", "0")
            in ("1", "true"),
            client_id=LINEAR_CLIENT_ID,
            client_secret=LINEAR_CLIENT_SECRET,
            web_host=WEB_HOST,
        )
        return LinearPlugin(config)
