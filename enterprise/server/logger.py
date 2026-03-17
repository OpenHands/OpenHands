import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

from pythonjsonlogger.json import JsonFormatter

from openhands.core.logger import openhands_logger

LOG_JSON = os.getenv('LOG_JSON', '1') == '1'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
DEBUG = os.getenv('DEBUG', 'False').lower() in ['true', '1', 'yes']
if DEBUG:
    LOG_LEVEL = 'DEBUG'

FILE_PREFIX = 'File "'
CWD_PREFIX = FILE_PREFIX + str(Path(os.getcwd()).parent) + '/'
SITE_PACKAGES_PREFIX = (
    CWD_PREFIX
    + f'.venv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/'
)
# Make the JSON easy to read in the console - useful for non cloud environments
LOG_JSON_FOR_CONSOLE = int(os.getenv('LOG_JSON_FOR_CONSOLE', '0'))


def format_stack(stack: str) -> list[str]:
    return (
        stack.replace(SITE_PACKAGES_PREFIX, FILE_PREFIX)
        .replace(CWD_PREFIX, FILE_PREFIX)
        .replace('"', "'")
        .split('\n')
    )


def custom_json_serializer(obj, **kwargs):
    if LOG_JSON_FOR_CONSOLE:
        # Format json output
        kwargs['indent'] = 2
        obj = {'ts': datetime.now().isoformat(), **obj}

        # Format stack traces
        if isinstance(obj, dict):
            exc_info = obj.get('exc_info')
            if isinstance(exc_info, str):
                obj['exc_info'] = format_stack(exc_info)
            stack_info = obj.get('stack_info')
            if isinstance(stack_info, str):
                obj['stack_info'] = format_stack(stack_info)

    result = json.dumps(obj, **kwargs)

    # Swap out newlines to make things easier to read. This will produce
    # invalid json but means we can have similar logs in local development
    # to production, making things easier to correlate. Obviously,
    # LOG_JSON_FOR_CONSOLE should not be used in production environments.
    if LOG_JSON_FOR_CONSOLE:
        result = result.replace('\\n', '\n')

    return result


def setup_json_logger(
    logger: logging.Logger,
    level: str = LOG_LEVEL,
    _out: TextIO = sys.stdout,
) -> None:
    """
    Configure logger instance to output json for Google Cloud.
    Existing filters should stay in place for sensitive content.
    """

    # Remove existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    handler = logging.StreamHandler(_out)
    handler.setLevel(level)

    formatter = JsonFormatter(
        '{message}{levelname}',
        style='{',
        rename_fields={'levelname': 'severity'},
        json_serializer=custom_json_serializer,
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)


def setup_all_loggers():
    """
    Setup JSON logging for all libraries that may be logging.
    Leave OpenHands alone since it's already configured.
    """
    if LOG_JSON:
        # Setup the root logger
        setup_json_logger(logging.getLogger())

        for name in logging.root.manager.loggerDict:
            logger = logging.getLogger(name)
            setup_json_logger(logger)
            logger.propagate = False

    # Quiet down some of the loggers that talk too much!
    loquacious_loggers = {
        'engineio',
        'engineio.server',
        'fastmcp',
        'FastMCP',
        'httpx',
        'mcp.client.sse',
        'socketio',
        'socketio.client',
        'socketio.server',
        'sqlalchemy.engine.Engine',
        'sqlalchemy.orm.mapper.Mapper',
    }
    for logger_name in loquacious_loggers:
        logging.getLogger(logger_name).setLevel('WARNING')


logger = logging.getLogger('saas')
setup_all_loggers()
# Openhands logger is heavily customized - so we want to make sure that it is logging json
setup_json_logger(openhands_logger)
