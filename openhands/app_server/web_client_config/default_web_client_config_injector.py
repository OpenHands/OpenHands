

from openhands.app_server.web_client_config.web_client_config_injector import WebClientConfigInjector
from openhands.app_server.web_client_config.web_client_config_models import WebClientConfig


class DefaultWebClientConfigInjector(WebClientConfigInjector, WebClientConfig):

    async def get_web_client_config(self) -> WebClientConfig:
        return WebClientConfig({
            key: getattr(self, key)
            for key in WebClientConfig.model_fields
        })
