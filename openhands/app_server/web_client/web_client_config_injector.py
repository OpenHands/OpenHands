from abc import abstractmethod

from openhands.agent_server.env_parser import ABC, DiscriminatedUnionMixin
from openhands.app_server.web_client.web_client_models import WebClientConfig


class WebClientConfigInjector(DiscriminatedUnionMixin, ABC):
    @abstractmethod
    async def get_web_client_config(self) -> WebClientConfig:
        """Get the current web client configuration."""
