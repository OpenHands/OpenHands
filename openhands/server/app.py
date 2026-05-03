# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
from openhands.server.optional_legacy_imports import (
    _import_optional_agenthub,
    _load_optional_security_api_router,
)

_import_optional_agenthub()

security_api_router = _load_optional_security_api_router()

# DEPRECATED: This module is deprecated and will be removed in a future release.
# Please use openhands.app_server.app instead.
#
# For backward compatibility, this module re-exports the app from openhands.app_server.app.
# Use openhands.server.listen or openhands.app_server.app directly for the fully configured
# application (middleware, static files, etc.).

from openhands.app_server.app import (
    app,
    authentication_error_handler,
    combine_lifespans,
    mcp_app,
)

if security_api_router is not None:
    app.include_router(security_api_router)

__all__ = ["app", "mcp_app", "combine_lifespans", "authentication_error_handler"]
