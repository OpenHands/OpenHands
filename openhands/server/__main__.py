import os
import warnings

import uvicorn

from openhands.core.logger import get_uvicorn_json_log_config


def main():
    # Suppress SyntaxWarnings from pydub.utils about invalid escape sequences
    warnings.filterwarnings('ignore', category=SyntaxWarning, module=r'pydub\.utils')

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
