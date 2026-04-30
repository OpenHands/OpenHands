# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.

from dotenv import load_dotenv

from openhands.app_server.secrets.secrets_store import SecretsStore
from openhands.app_server.settings.settings_store import SettingsStore
from openhands.app_server.utils.import_utils import get_impl
from openhands.core.config import load_openhands_config
from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.server.config.server_config import ServerConfig, load_server_config
from openhands.server.types import ServerConfigInterface

load_dotenv()

config: OpenHandsConfig = load_openhands_config()
server_config_interface: ServerConfigInterface = load_server_config()
assert isinstance(server_config_interface, ServerConfig), (
    'Loaded server config interface is not a ServerConfig, despite this being assumed'
)
server_config: ServerConfig = server_config_interface

# Note: socketio is no longer used. Redis access should use the standard redis package directly.
# For enterprise code, use: from enterprise.storage.redis import get_redis_client, get_redis_client_async

SettingsStoreImpl = get_impl(SettingsStore, server_config.settings_store_class)

SecretsStoreImpl = get_impl(SecretsStore, server_config.secret_store_class)
