# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
import os
import secrets

import uvicorn

from openhands.core.logger import get_uvicorn_json_log_config


def _ensure_session_api_key() -> None:
    """Auto-generate a SESSION_API_KEY if one is not set and no explicit opt-out is configured.

    This must be called BEFORE uvicorn imports the application modules, since several modules
    read SESSION_API_KEY from os.environ at import time.
    """
    if os.environ.get('SESSION_API_KEY'):
        return

    # Explicit opt-out for local development
    if os.environ.get('OPENHANDS_NO_AUTH', '').lower() in ('1', 'true'):
        print(
            '\n'
            '=' * 70 + '\n'
            'WARNING: Authentication is DISABLED (OPENHANDS_NO_AUTH=1).\n'
            'All API and WebSocket endpoints are accessible without a key.\n'
            'Do NOT use this setting in production or on a public network.\n'
            '=' * 70 + '\n'
        )
        return

    # Check if a custom auth class is configured (implies external auth handling)
    config_cls = os.environ.get('OPENHANDS_CONFIG_CLS')
    if config_cls:
        return

    # Auto-generate a random key
    generated_key = secrets.token_hex(32)
    os.environ['SESSION_API_KEY'] = generated_key
    print(
        '\n'
        '=' * 70 + '\n'
        'No SESSION_API_KEY set. Auto-generated a session API key.\n'
        'Use this key for API/WebSocket access:\n'
        f'\n  SESSION_API_KEY={generated_key}\n\n'
        'To disable authentication (local dev only), set OPENHANDS_NO_AUTH=1\n'
        'To use your own key, set the SESSION_API_KEY environment variable.\n'
        '=' * 70 + '\n'
    )


def main():
    _ensure_session_api_key()

    # When LOG_JSON=1, configure Uvicorn to emit JSON logs for error/access
    log_config = None
    if os.getenv('LOG_JSON', '0') in ('1', 'true', 'True'):
        log_config = get_uvicorn_json_log_config()

    uvicorn.run(
        'openhands.server.listen:app',
        host='0.0.0.0',
        port=int(os.environ.get('port') or '3000'),
        log_level='debug' if os.environ.get('DEBUG') else 'info',
        log_config=log_config,
        # If LOG_JSON enabled, force colors off; otherwise let uvicorn default
        use_colors=False if log_config else None,
    )


if __name__ == '__main__':
    main()
