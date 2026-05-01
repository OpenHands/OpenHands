# DEPRECATED: This module is deprecated and will be removed in a future release.
# Please use openhands.app_server.app instead.
#
# For backward compatibility, this module re-exports the app from openhands.app_server.app.
# Note: This module does NOT include middleware setup. Use openhands.server.listen or
# openhands.app_server.app directly for the fully configured application.

from openhands.app_server.app import (
    app,
    authentication_error_handler,
    combine_lifespans,
    mcp_app,
)

__all__ = ['app', 'mcp_app', 'combine_lifespans', 'authentication_error_handler']
