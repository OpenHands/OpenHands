# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import copy
import logging
import os
import re
import sys
import traceback
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from types import TracebackType
from typing import Any, Literal, Mapping, MutableMapping, TextIO

import litellm
from pythonjsonlogger.json import JsonFormatter
from termcolor import colored

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
DEBUG = os.getenv('DEBUG', 'False').lower() in ['true', '1', 'yes']
DEBUG_LLM = os.getenv('DEBUG_LLM', 'False').lower() in ['true', '1', 'yes']

# Structured logs with JSON, disabled by default
LOG_JSON = os.getenv('LOG_JSON', 'False').lower() in ['true', '1', 'yes']
LOG_JSON_LEVEL_KEY = os.getenv('LOG_JSON_LEVEL_KEY', 'level')


# Configure litellm logging based on DEBUG_LLM
if DEBUG_LLM:
    confirmation = input(
        '\n⚠️ WARNING: You are enabling DEBUG_LLM which may expose sensitive information like API keys.\n'
        'This should NEVER be enabled in production.\n'
        "Type 'y' to confirm you understand the risks: "
    )
    if confirmation.lower() == 'y':
        litellm.suppress_debug_info = False
        litellm.set_verbose = True
    else:
        print('DEBUG_LLM disabled due to lack of confirmation')
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
else:
    litellm.suppress_debug_info = True
    litellm.set_verbose = False

if DEBUG:
    LOG_LEVEL = 'DEBUG'

LOG_TO_FILE = os.getenv('LOG_TO_FILE', str(LOG_LEVEL == 'DEBUG')).lower() in [
    'true',
    '1',
    'yes',
]
DISABLE_COLOR_PRINTING = False

LOG_ALL_EVENTS = os.getenv('LOG_ALL_EVENTS', 'False').lower() in ['true', '1', 'yes']

# Controls whether to stream Docker container logs
DEBUG_RUNTIME = os.getenv('DEBUG_RUNTIME', 'False').lower() in ['true', '1', 'yes']

ColorType = Literal[
    'red',
    'green',
    'yellow',
    'blue',
    'magenta',
    'cyan',
    'light_grey',
    'dark_grey',
    'light_red',
    'light_green',
    'light_yellow',
    'light_blue',
    'light_magenta',
    'light_cyan',
    'white',
]

LOG_COLORS: Mapping[str, ColorType] = {
    'ACTION': 'green',
    'USER_ACTION': 'light_red',
    'OBSERVATION': 'yellow',
    'USER_OBSERVATION': 'light_green',
    'DETAIL': 'cyan',
    'ERROR': 'red',
    'PLAN': 'light_magenta',
}


class StackInfoFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            # Only add stack trace info if there's an actual exception
            exc_info = sys.exc_info()
            if exc_info and exc_info[0] is not None:
                # Capture the current stack trace as a string
                stack = traceback.format_stack()
                # Remove the last entries which are related to the logging machinery
                stack = stack[:-3]  # Adjust this number if needed
                # Join the stack frames into a single string
                stack_str = ''.join(stack)
                setattr(record, 'stack_info', stack_str)
                setattr(record, 'exc_info', exc_info)
        return True


class NoColorFormatter(logging.Formatter):
    """Formatter for non-colored logging in files."""

    def format(self, record: logging.LogRecord) -> str:
        # Create a deep copy of the record to avoid modifying the original
        new_record = _fix_record(record)

        # Strip ANSI color codes from the message
        new_record.msg = strip_ansi(new_record.msg)

        return super().format(new_record)


def strip_ansi(s: str) -> str:
    """Remove ANSI escape sequences (terminal color/formatting codes) from string.

    Removes ANSI escape sequences from str, as defined by ECMA-048 in
    http://www.ecma-international.org/publications/files/ECMA-ST/Ecma-048.pdf
    # https://github.com/ewen-lbh/python-strip-ansi/blob/master/strip_ansi/__init__.py
    """
    pattern = re.compile(r'\x1B\[\d+(;\d+){0,2}m')
    stripped = pattern.sub('', s)
    return stripped


class ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg_type = record.__dict__.get('msg_type', '')
        event_source = record.__dict__.get('event_source', '')
        if event_source:
            new_msg_type = f'{event_source.upper()}_{msg_type}'
            if new_msg_type in LOG_COLORS:
                msg_type = new_msg_type
        if msg_type in LOG_COLORS and not DISABLE_COLOR_PRINTING:
            msg_type_color = colored(msg_type, LOG_COLORS[msg_type])
            msg = colored(record.msg, LOG_COLORS[msg_type])
            time_str = colored(
                self.formatTime(record, self.datefmt), LOG_COLORS[msg_type]
            )
            name_str = colored(record.name, LOG_COLORS[msg_type])
            level_str = colored(record.levelname, LOG_COLORS[msg_type])
            if msg_type in ['ERROR'] or DEBUG:
                return f'{time_str} - {name_str}:{level_str}: {record.filename}:{record.lineno}\n{msg_type_color}\n{msg}'
            return f'{time_str} - {msg_type_color}\n{msg}'
        elif msg_type == 'STEP':
            if LOG_ALL_EVENTS:
                msg = '\n\n==============\n' + record.msg + '\n'
                return f'{msg}'
            else:
                return record.msg

        new_record = _fix_record(record)
        return super().format(new_record)


def _fix_record(record: logging.LogRecord) -> logging.LogRecord:
    new_record = copy.copy(record)
    # The formatter expects non boolean values, and will raise an exception if there is a boolean - so we fix these
    # LogRecord attributes are dynamically typed
    if getattr(new_record, 'exc_info', None) is True:
        setattr(new_record, 'exc_info', sys.exc_info())
        setattr(new_record, 'stack_info', None)
    return new_record


file_formatter = NoColorFormatter(
    '%(asctime)s - %(name)s:%(levelname)s: %(filename)s:%(lineno)s - %(message)s',
    datefmt='%H:%M:%S',
)
llm_formatter = logging.Formatter('%(message)s')


class RollingLogger:
    max_lines: int
    char_limit: int
    log_lines: list[str]
    all_lines: str

    def __init__(self, max_lines: int = 10, char_limit: int = 80) -> None:
        self.max_lines = max_lines
        self.char_limit = char_limit
        self.log_lines = [''] * self.max_lines
        self.all_lines = ''

    def is_enabled(self) -> bool:
        return DEBUG and sys.stdout.isatty()

    def start(self, message: str = '') -> None:
        if message:
            print(message)
        self._write('\n' * self.max_lines)
        self._flush()

    def add_line(self, line: str) -> None:
        self.log_lines.pop(0)
        self.log_lines.append(line[: self.char_limit])
        self.print_lines()
        self.all_lines += line + '\n'

    def write_immediately(self, line: str) -> None:
        self._write(line)
        self._flush()

    def print_lines(self) -> None:
        """Display the last n log_lines in the console (not for file logging).

        This will create the effect of a rolling display in the console.
        """
        self.move_back()
        for line in self.log_lines:
            self.replace_current_line(line)

    def move_back(self, amount: int = -1) -> None:
        r"""'\033[F' moves the cursor up one line."""
        if amount == -1:
            amount = self.max_lines
        self._write('\033[F' * amount)
        self._flush()

    def replace_current_line(self, line: str = '') -> None:
        r"""'\033[2K\r' clears the line and moves the cursor to the beginning of the line."""
        self._write('\033[2K' + line + '\n')
        self._flush()

    def _write(self, line: str) -> None:
        if not self.is_enabled():
            return
        sys.stdout.write(line)

    def _flush(self) -> None:
        if not self.is_enabled():
            return
        sys.stdout.flush()


class SensitiveDataFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Gather sensitive values which should not ever appear in the logs.
        sensitive_values = []
        for key, value in os.environ.items():
            key_upper = key.upper()
            if (
                len(value) > 2
                and value != 'default'
                and any(s in key_upper for s in ('SECRET', '_KEY', '_CODE', '_TOKEN'))
            ):
                sensitive_values.append(value)

        # Replace sensitive values from env!
        msg = record.getMessage()
        for sensitive_value in sensitive_values:
            msg = msg.replace(sensitive_value, '******')

        # Replace obvious sensitive values from log itself...
        sensitive_patterns = [
            'api_key',
            'aws_access_key_id',
            'aws_secret_access_key',
            'e2b_api_key',
            'github_token',
            'jwt_secret',
            'modal_api_token_id',
            'modal_api_token_secret',
            'llm_api_key',
            'sandbox_env_github_token',
            'runloop_api_key',
            'daytona_api_key',
        ]

        # add env var names
        env_vars = [attr.upper() for attr in sensitive_patterns]
        sensitive_patterns.extend(env_vars)

        for attr in sensitive_patterns:
            pattern = rf"{attr}='?([\w-]+)'?"
            msg = re.sub(pattern, f"{attr}='******'", msg)

        # Update the record
        record.msg = msg
        record.args = ()

        return True


def get_console_handler(log_level: int = logging.INFO) -> logging.StreamHandler:
    """Returns a console handler for logging."""
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter_str = '\033[92m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s'
    console_handler.setFormatter(ColoredFormatter(formatter_str, datefmt='%H:%M:%S'))
    return console_handler


def get_file_handler(
    log_dir: str,
    log_level: int = logging.INFO,
    when: str = 'd',
    backup_count: int = 7,
    utc: bool = False,
) -> TimedRotatingFileHandler:
    """Returns a file handler for logging."""
    os.makedirs(log_dir, exist_ok=True)
    file_name = 'openhands.log'
    file_handler = TimedRotatingFileHandler(
        os.path.join(log_dir, file_name),
        when=when,
        backupCount=backup_count,
        utc=utc,
    )
    file_handler.setLevel(log_level)
    if LOG_JSON:
        file_handler.setFormatter(json_formatter())
    else:
        file_handler.setFormatter(file_formatter)
    return file_handler


def json_formatter() -> JsonFormatter:
    return JsonFormatter(
        '{message}{levelname}',
        style='{',
        rename_fields={'levelname': LOG_JSON_LEVEL_KEY},
        timestamp=True,
    )


def json_log_handler(
    level: int = logging.INFO,
    _out: TextIO = sys.stdout,
) -> logging.Handler:
    """Configure logger instance for structured logging as json lines."""
    handler = logging.StreamHandler(_out)
    handler.setLevel(level)
    handler.setFormatter(json_formatter())
    return handler


# Set up logging
logging.basicConfig(level=logging.ERROR)


def log_uncaught_exceptions(
    ex_cls: type[BaseException], ex: BaseException, tb: TracebackType | None
) -> Any:
    """Logs uncaught exceptions in structured form when JSON logging is enabled.

    Args:
        ex_cls: The type of the exception.
        ex: The exception instance.
        tb: The traceback object.

    Returns:
        None
    """
    # Route uncaught exceptions through our logger with proper exc_info
    # Avoid manual formatting which creates multi-line plain-text log entries
    openhands_logger.error('Uncaught exception', exc_info=(ex_cls, ex, tb))


sys.excepthook = log_uncaught_exceptions
openhands_logger = logging.getLogger('openhands')
current_log_level = logging.INFO

if LOG_LEVEL in logging.getLevelNamesMapping():
    current_log_level = logging.getLevelNamesMapping()[LOG_LEVEL]
openhands_logger.setLevel(current_log_level)

if DEBUG:
    openhands_logger.addFilter(StackInfoFilter())

if current_log_level == logging.DEBUG:
    openhands_logger.debug('DEBUG mode enabled.')

if LOG_JSON:
    openhands_logger.addHandler(json_log_handler(current_log_level))
    # Configure concurrent.futures logger to use JSON formatting as well
    cf_logger = logging.getLogger('concurrent.futures')
    cf_logger.setLevel(current_log_level)
    cf_logger.addHandler(json_log_handler(current_log_level))
else:
    openhands_logger.addHandler(get_console_handler(current_log_level))

openhands_logger.addFilter(SensitiveDataFilter(openhands_logger.name))
openhands_logger.propagate = False
openhands_logger.debug('Logging initialized')

LOG_DIR = os.path.join(
    # parent dir of openhands/core (i.e., root of the repo)
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'logs',
)

if LOG_TO_FILE:
    openhands_logger.addHandler(
        get_file_handler(LOG_DIR, current_log_level)
    )  # default log to project root
    openhands_logger.debug(f'Logging to file in: {LOG_DIR}')

# Exclude LiteLLM from logging output as it can leak keys
logging.getLogger('LiteLLM').disabled = True
logging.getLogger('LiteLLM Router').disabled = True
logging.getLogger('LiteLLM Proxy').disabled = True

# Exclude loquacious loggers
LOQUACIOUS_LOGGERS = [
    'engineio',
    'engineio.server',
    'socketio',
    'socketio.client',
    'socketio.server',
    'aiosqlite',
]

for logger_name in LOQUACIOUS_LOGGERS:
    logging.getLogger(logger_name).setLevel('WARNING')


class LlmFileHandler(logging.FileHandler):
    """LLM prompt and response logging."""

    def __init__(
        self,
        filename: str,
        mode: str = 'a',
        encoding: str = 'utf-8',
        delay: bool = False,
    ) -> None:
        """Initializes an instance of LlmFileHandler.

        Args:
            filename (str): The name of the log file.
            mode (str, optional): The file mode. Defaults to 'a'.
            encoding (str, optional): The file encoding. Defaults to None.
            delay (bool, optional): Whether to delay file opening. Defaults to False.
        """
        self.filename = filename
        self.message_counter = 1
        if DEBUG:
            self.session = datetime.now().strftime('%y-%m-%d_%H-%M')
        else:
            self.session = 'default'
        self.log_directory = os.path.join(LOG_DIR, 'llm', self.session)
        os.makedirs(self.log_directory, exist_ok=True)
        if not DEBUG:
            # Clear the log directory if not in debug mode
            for file in os.listdir(self.log_directory):
                file_path = os.path.join(self.log_directory, file)
                try:
                    os.unlink(file_path)
                except Exception as e:
                    openhands_logger.error(
                        'Failed to delete %s. Reason: %s', file_path, e
                    )
        filename = f'{self.filename}_{self.message_counter:03}.log'
        self.baseFilename = os.path.join(self.log_directory, filename)
        super().__init__(self.baseFilename, mode, encoding, delay)

    def emit(self, record: logging.LogRecord) -> None:
        """Emits a log record.

        Args:
            record (logging.LogRecord): The log record to emit.
        """
        filename = f'{self.filename}_{self.message_counter:03}.log'
        self.baseFilename = os.path.join(self.log_directory, filename)
        self.stream = self._open()
        super().emit(record)
        self.stream.close()
        openhands_logger.debug('Logging to %s', self.baseFilename)
        self.message_counter += 1


def _get_llm_file_handler(name: str, log_level: int) -> LlmFileHandler:
    # The 'delay' parameter, when set to True, postpones the opening of the log file
    # until the first log message is emitted.
    llm_file_handler = LlmFileHandler(name, delay=True)
    llm_file_handler.setFormatter(llm_formatter)
    llm_file_handler.setLevel(log_level)
    return llm_file_handler


def _setup_llm_logger(name: str, log_level: int) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(log_level)
    if LOG_TO_FILE:
        logger.addHandler(_get_llm_file_handler(name, log_level))
    return logger


llm_prompt_logger = _setup_llm_logger('prompt', current_log_level)
llm_response_logger = _setup_llm_logger('response', current_log_level)


class OpenHandsLoggerAdapter(logging.LoggerAdapter):
    extra: dict

    def __init__(
        self, logger: logging.Logger = openhands_logger, extra: dict | None = None
    ) -> None:
        self.logger = logger
        self.extra = extra or {}

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        """If 'extra' is supplied in kwargs, merge it with the adapters 'extra' dict
        Starting in Python 3.13, LoggerAdapter's merge_extra option will do this.
        """
        if 'extra' in kwargs and isinstance(kwargs['extra'], dict):
            kwargs['extra'] = {**self.extra, **kwargs['extra']}
        else:
            kwargs['extra'] = self.extra
        return msg, kwargs


def get_uvicorn_json_log_config() -> dict:
    """Returns a uvicorn log config dict for JSON structured logging.

    This configuration ensures Uvicorn's error and access logs are emitted
    as single-line JSON when LOG_JSON=1, avoiding multi-line plain-text
    tracebacks in log aggregators like Datadog.

    Returns:
        A dict suitable for passing to uvicorn.run(..., log_config=...).
    """
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            # Uvicorn mutates 'default' and 'access' to set use_colors;
            # keep them present using Uvicorn formatters
            'default': {
                '()': 'uvicorn.logging.DefaultFormatter',
                'fmt': '%(levelprefix)s %(message)s',
                'use_colors': None,
            },
            'access': {
                '()': 'uvicorn.logging.AccessFormatter',
                'fmt': '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
                'use_colors': None,
            },
            # Actual JSON formatters used by handlers below
            'json': {
                '()': 'pythonjsonlogger.json.JsonFormatter',
                'fmt': '%(message)s %(levelname)s %(name)s %(asctime)s %(exc_info)s',
            },
            'json_access': {
                '()': 'pythonjsonlogger.json.JsonFormatter',
                'fmt': '%(message)s %(levelname)s %(name)s %(asctime)s %(client_addr)s %(request_line)s %(status_code)s',
            },
        },
        'handlers': {
            'default': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'json',
                'stream': 'ext://sys.stdout',
            },
            'access': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'json_access',
                'stream': 'ext://sys.stdout',
            },
        },
        'loggers': {
            'uvicorn': {
                'handlers': ['default'],
                'level': 'INFO',
                'propagate': False,
            },
            'uvicorn.error': {
                'handlers': ['default'],
                'level': 'INFO',
                'propagate': False,
            },
            'uvicorn.access': {
                'handlers': ['access'],
                'level': 'INFO',
                'propagate': False,
            },
            # Suppress LiteLLM loggers to prevent them from leaking through root logger
            # This is necessary because logging.config.dictConfig() resets the .disabled flag
            'LiteLLM': {
                'handlers': [],
                'level': 'CRITICAL',
                'propagate': False,
            },
            'LiteLLM Router': {
                'handlers': [],
                'level': 'CRITICAL',
                'propagate': False,
            },
            'LiteLLM Proxy': {
                'handlers': [],
                'level': 'CRITICAL',
                'propagate': False,
            },
        },
        'root': {'level': 'INFO', 'handlers': ['default']},
    }
