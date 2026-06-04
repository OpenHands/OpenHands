# DEPRECATED: This module is deprecated. Use uvicorn directly:
#   uvicorn openhands.app_server.app:app --host 0.0.0.0 --port 3000
#
# This module is kept for backward compatibility.

import os

import uvicorn

from openhands.app_server.utils.logger import LOG_JSON, get_uvicorn_log_config


def main():
    log_config = get_uvicorn_log_config()
    port = os.environ.get('PORT') or os.environ.get('port') or '3000'

    uvicorn.run(
        'openhands.server.listen:app',
        host='0.0.0.0',
        port=int(port),
        log_level='debug' if os.environ.get('DEBUG') else 'info',
        log_config=log_config,
        use_colors=False if LOG_JSON else None,
    )


if __name__ == '__main__':
    main()
