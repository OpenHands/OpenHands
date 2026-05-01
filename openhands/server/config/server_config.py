# DEPRECATED: This module is deprecated and will be removed in a future release.
# Please use openhands.app_server.server_config.server_config instead.
#
# For backward compatibility, this module re-exports from openhands.app_server.server_config.server_config.

from openhands.app_server.server_config.server_config import (
    ServerConfig,
    load_server_config,
)

__all__ = ['ServerConfig', 'load_server_config']
