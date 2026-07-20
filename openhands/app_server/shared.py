from dotenv import load_dotenv

from openhands.app_server.secrets.secrets_store import SecretsStore
from openhands.app_server.server_config.server_config import (
    ServerConfig,
    load_server_config,
)
from openhands.app_server.settings.settings_store import SettingsStore
from openhands.app_server.types import ServerConfigInterface
from openhands.app_server.utils.import_utils import get_impl

load_dotenv()

server_config_interface: ServerConfigInterface = load_server_config()
assert isinstance(server_config_interface, ServerConfig), (
    'Loaded server config interface is not a ServerConfig, despite this being assumed'
)
server_config: ServerConfig = server_config_interface

# Note: socketio is no longer used. Redis access should use the standard redis package directly.
# For enterprise code, use: from enterprise.storage.redis import get_redis_client, get_redis_client_async

SettingsStoreImpl = get_impl(SettingsStore, server_config.settings_store_class)

SecretsStoreImpl = get_impl(SecretsStore, server_config.secret_store_class)
