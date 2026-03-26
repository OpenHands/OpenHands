import json
import logging
import os
from io import StringIO
from unittest.mock import patch

import pytest
from freezegun import freeze_time
from server.logger import format_stack, setup_json_logger

from openhands.core.logger import openhands_logger

FROZEN_TIMESTAMP = '2024-01-15T10:30:00+00:00'
# datetime.now().isoformat() doesn't include timezone info
FROZEN_TIMESTAMP_NO_TZ = '2024-01-15T10:30:00'


@pytest.fixture
def log_output():
    """Fixture to capture log output"""
    string_io = StringIO()
    logger = logging.Logger('test')
    setup_json_logger(logger, 'INFO', _out=string_io)

    return logger, string_io


class TestLogOutput:
    @freeze_time(FROZEN_TIMESTAMP)
    def test_info(self, log_output):
        logger, string_io = log_output

        logger.info('Test message')
        output = json.loads(string_io.getvalue())
        assert output['message'] == 'Test message'
        assert output['severity'] == 'INFO'
        assert output['ts'] == FROZEN_TIMESTAMP
        assert output['module'] == 'test_logger'
        assert output['funcName'] == 'test_info'
        assert 'lineno' in output

    @freeze_time(FROZEN_TIMESTAMP)
    def test_error(self, log_output):
        logger, string_io = log_output

        logger.error('Test message')
        output = json.loads(string_io.getvalue())
        assert output['message'] == 'Test message'
        assert output['severity'] == 'ERROR'
        assert output['ts'] == FROZEN_TIMESTAMP
        assert output['module'] == 'test_logger'
        assert output['funcName'] == 'test_error'
        assert 'lineno' in output

    @freeze_time(FROZEN_TIMESTAMP)
    def test_extra_fields(self, log_output):
        logger, string_io = log_output

        logger.info('Test message', extra={'key': '..val..'})
        output = json.loads(string_io.getvalue())
        assert output['key'] == '..val..'
        assert output['message'] == 'Test message'
        assert output['severity'] == 'INFO'
        assert output['ts'] == FROZEN_TIMESTAMP
        assert output['module'] == 'test_logger'
        assert output['funcName'] == 'test_extra_fields'
        assert 'lineno' in output

    def test_format_stack(self):
        stack = (
            '"  + Exception Group Traceback (most recent call last):\n'
            ''
            '  |   File "/app/.venv/lib/python3.12/site-packages/starlette/_utils.py", line 76, in collapse_excgroups\n'
            '  |     yield\n'
            '  |   File "/app/.venv/lib/python3.12/site-packages/starlette/middleware/base.py", line 174, in __call__\n'
            '  |     async with anyio.create_task_group() as task_group:\n'
            '  |   File "/app/.venv/lib/python3.12/site-packages/anyio/_backends/_asyncio.py", line 772, in __aexit__\n'
            '  |     raise BaseExceptionGroup(\n'
            '  | ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)\n'
            '  +-+---------------- 1 ----------------\n'
            '    | Traceback (most recent call last):\n'
            '    |   File "/app/.venv/lib/python3.12/site-packages/uvicorn/protocols/http/h11_impl.py", line 403, in run_asgi\n'
            '    |     result = await app(  # type: ignore[func-returns-value]\n'
            '    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '    |   File "/app/.venv/lib/python3.12/site-packages/uvicorn/middleware/proxy_headers.py", line 60, in __call__\n'
            '    |     return await self.app(scope, receive, send)\n'
            '    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '    |   File "/app/.venv/lib/python3.12/site-packages/engineio/async_drivers/asgi.py", line 75, in __call__\n'
            '    |     await self.other_asgi_app(scope, receive, send)\n'
            '    |   File "/app/.venv/lib/python3.12/site-packages/fastapi/applications.py", line 1054, in __call__\n'
            '    |     await super().__call__(scope, receive, send)\n'
            '    |   File "/app/.venv/lib/python3.12/site-packages/starlette/applications.py", line 112, in __call__\n'
            '    |     await self.middleware_stack(scope, receive, send)\n'
            '    |   File "/app/.venv/lib/python3.12/site-packages/starlette/middleware/errors.py", line 187, in __call__\n'
            '    |     raise exc\n'
            '    |   File "/app/.venv/lib/python3.12/site-packages/starlette/middleware/errors.py", line 165, in __call__\n'
            '    |     await self.app(scope, receive, _send)\n'
            '    |   File "/app/.venv/lib/python3.12/site-packages/starlette/middleware/base.py", line 173, in __call__\n'
            '    |     with recv_stream, send_stream, collapse_excgroups():\n'
            '    |   File "/usr/local/lib/python3.12/contextlib.py", line 158, in __exit__\n'
            '    |     self.gen.throw(value)\n'
            '    |   File "/app/.venv/lib/python3.12/site-packages/starlette/_utils.py", line 82, in collapse_excgroups\n'
            '    |     raise exc\n'
            '    |   File "/app/.venv/lib/python3.12/site-packages/starlette/middleware/base.py", line 175, in __call__\n'
            '    |     response = await self.dispatch_func(request, call_next)\n'
            '    |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '    |   File "/app/server/middleware.py", line 66, in __call__\n'
            '    |     self._check_tos(request)\n'
            '    |   File "/app/server/middleware.py", line 110, in _check_tos\n'
            '    |     decoded = jwt.decode(\n'
            '    |               ^^^^^^^^^^^\n'
            '    |   File "/app/.venv/lib/python3.12/site-packages/jwt/api_jwt.py", line 222, in decode\n'
            '    |     decoded = self.decode_complete(\n'
            '    |               ^^^^^^^^^^^^^^^^^^^^^\n'
            '    |   File "/app/.venv/lib/python3.12/site-packages/jwt/api_jwt.py", line 156, in decode_complete\n'
            '    |     decoded = api_jws.decode_complete(\n'
            '    |               ^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '    |   File "/app/.venv/lib/python3.12/site-packages/jwt/api_jws.py", line 220, in decode_complete\n'
            '    |     self._verify_signature(signing_input, header, signature, key, algorithms)\n'
            '    |   File "/app/.venv/lib/python3.12/site-packages/jwt/api_jws.py", line 328, in _verify_signature\n'
            '    |     raise InvalidSignatureError("Signature verification failed")\n'
            '    | jwt.exceptions.InvalidSignatureError: Signature verification failed\n'
            '    +------------------------------------\n'
            '\n'
            'During handling of the above exception, another exception occurred:\n'
            '\n'
            'Traceback (most recent call last):\n'
            '  File "/app/.venv/lib/python3.12/site-packages/uvicorn/protocols/http/h11_impl.py", line 403, in run_asgi\n'
            '    result = await app(  # type: ignore[func-returns-value]\n'
            '             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/app/.venv/lib/python3.12/site-packages/uvicorn/middleware/proxy_headers.py", line 60, in __call__\n'
            '    return await self.app(scope, receive, send)\n'
            '           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/app/.venv/lib/python3.12/site-packages/engineio/async_drivers/asgi.py", line 75, in __call__\n'
            '    await self.other_asgi_app(scope, receive, send)\n'
            '  File "/app/.venv/lib/python3.12/site-packages/fastapi/applications.py", line 1054, in __call__\n'
            '    await super().__call__(scope, receive, send)\n'
            '  File "/app/.venv/lib/python3.12/site-packages/starlette/applications.py", line 112, in __call__\n'
            '    await self.middleware_stack(scope, receive, send)\n'
            '  File "/app/.venv/lib/python3.12/site-packages/starlette/middleware/errors.py", line 187, in __call__\n'
            '    raise exc\n'
            '  File "/app/.venv/lib/python3.12/site-packages/starlette/middleware/errors.py", line 165, in __call__\n'
            '    await self.app(scope, receive, _send)\n'
            '  File "/app/.venv/lib/python3.12/site-packages/starlette/middleware/base.py", line 173, in __call__\n'
            '    with recv_stream, send_stream, collapse_excgroups():\n'
            '  File "/usr/local/lib/python3.12/contextlib.py", line 158, in __exit__\n'
            '    self.gen.throw(value)\n'
            '  File "/app/.venv/lib/python3.12/site-packages/starlette/_utils.py", line 82, in collapse_excgroups\n'
            '    raise exc\n'
            '  File "/app/.venv/lib/python3.12/site-packages/starlette/middleware/base.py", line 175, in __call__\n'
            '    response = await self.dispatch_func(request, call_next)\n'
            '               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/app/server/middleware.py", line 66, in __call__\n'
            '    self._check_tos(request)\n'
            '  File "/app/server/middleware.py", line 110, in _check_tos\n'
            '    decoded = jwt.decode(\n'
            '              ^^^^^^^^^^^\n'
            '  File "/app/.venv/lib/python3.12/site-packages/jwt/api_jwt.py", line 222, in decode\n'
            '    decoded = self.decode_complete(\n'
            '              ^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/app/.venv/lib/python3.12/site-packages/jwt/api_jwt.py", line 156, in decode_complete\n'
            '    decoded = api_jws.decode_complete(\n'
            '              ^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/app/.venv/lib/python3.12/site-packages/jwt/api_jws.py", line 220, in decode_complete\n'
            '    self._verify_signature(signing_input, header, signature, key, algorithms)\n'
            '  File "/app/.venv/lib/python3.12/site-packages/jwt/api_jws.py", line 328, in _verify_signature\n'
            '    raise InvalidSignatureError("Signature verification failed")\n'
            'jwt.exceptions.InvalidSignatureError: Signature verification failed"'
        )
        with (
            patch('server.logger.LOG_JSON_FOR_CONSOLE', 1),
            patch('server.logger.CWD_PREFIX', 'File "/app/'),
            patch(
                'server.logger.SITE_PACKAGES_PREFIX',
                'File "/app/.venv/lib/python3.12/site-packages/',
            ),
        ):
            formatted = format_stack(stack)
            expected = [
                "'  + Exception Group Traceback (most recent call last):",
                "  |   File 'starlette/_utils.py', line 76, in collapse_excgroups",
                '  |     yield',
                "  |   File 'starlette/middleware/base.py', line 174, in __call__",
                '  |     async with anyio.create_task_group() as task_group:',
                "  |   File 'anyio/_backends/_asyncio.py', line 772, in __aexit__",
                '  |     raise BaseExceptionGroup(',
                '  | ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)',
                '  +-+---------------- 1 ----------------',
                '    | Traceback (most recent call last):',
                "    |   File 'uvicorn/protocols/http/h11_impl.py', line 403, in run_asgi",
                '    |     result = await app(  # type: ignore[func-returns-value]',
                '    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^',
                "    |   File 'uvicorn/middleware/proxy_headers.py', line 60, in __call__",
                '    |     return await self.app(scope, receive, send)',
                '    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^',
                "    |   File 'engineio/async_drivers/asgi.py', line 75, in __call__",
                '    |     await self.other_asgi_app(scope, receive, send)',
                "    |   File 'fastapi/applications.py', line 1054, in __call__",
                '    |     await super().__call__(scope, receive, send)',
                "    |   File 'starlette/applications.py', line 112, in __call__",
                '    |     await self.middleware_stack(scope, receive, send)',
                "    |   File 'starlette/middleware/errors.py', line 187, in __call__",
                '    |     raise exc',
                "    |   File 'starlette/middleware/errors.py', line 165, in __call__",
                '    |     await self.app(scope, receive, _send)',
                "    |   File 'starlette/middleware/base.py', line 173, in __call__",
                '    |     with recv_stream, send_stream, collapse_excgroups():',
                "    |   File '/usr/local/lib/python3.12/contextlib.py', line 158, in __exit__",
                '    |     self.gen.throw(value)',
                "    |   File 'starlette/_utils.py', line 82, in collapse_excgroups",
                '    |     raise exc',
                "    |   File 'starlette/middleware/base.py', line 175, in __call__",
                '    |     response = await self.dispatch_func(request, call_next)',
                '    |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^',
                "    |   File 'server/middleware.py', line 66, in __call__",
                '    |     self._check_tos(request)',
                "    |   File 'server/middleware.py', line 110, in _check_tos",
                '    |     decoded = jwt.decode(',
                '    |               ^^^^^^^^^^^',
                "    |   File 'jwt/api_jwt.py', line 222, in decode",
                '    |     decoded = self.decode_complete(',
                '    |               ^^^^^^^^^^^^^^^^^^^^^',
                "    |   File 'jwt/api_jwt.py', line 156, in decode_complete",
                '    |     decoded = api_jws.decode_complete(',
                '    |               ^^^^^^^^^^^^^^^^^^^^^^^^',
                "    |   File 'jwt/api_jws.py', line 220, in decode_complete",
                '    |     self._verify_signature(signing_input, header, signature, key, algorithms)',
                "    |   File 'jwt/api_jws.py', line 328, in _verify_signature",
                "    |     raise InvalidSignatureError('Signature verification failed')",
                '    | jwt.exceptions.InvalidSignatureError: Signature verification failed',
                '    +------------------------------------',
                '',
                'During handling of the above exception, another exception occurred:',
                '',
                'Traceback (most recent call last):',
                "  File 'uvicorn/protocols/http/h11_impl.py', line 403, in run_asgi",
                '    result = await app(  # type: ignore[func-returns-value]',
                '             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^',
                "  File 'uvicorn/middleware/proxy_headers.py', line 60, in __call__",
                '    return await self.app(scope, receive, send)',
                '           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^',
                "  File 'engineio/async_drivers/asgi.py', line 75, in __call__",
                '    await self.other_asgi_app(scope, receive, send)',
                "  File 'fastapi/applications.py', line 1054, in __call__",
                '    await super().__call__(scope, receive, send)',
                "  File 'starlette/applications.py', line 112, in __call__",
                '    await self.middleware_stack(scope, receive, send)',
                "  File 'starlette/middleware/errors.py', line 187, in __call__",
                '    raise exc',
                "  File 'starlette/middleware/errors.py', line 165, in __call__",
                '    await self.app(scope, receive, _send)',
                "  File 'starlette/middleware/base.py', line 173, in __call__",
                '    with recv_stream, send_stream, collapse_excgroups():',
                "  File '/usr/local/lib/python3.12/contextlib.py', line 158, in __exit__",
                '    self.gen.throw(value)',
                "  File 'starlette/_utils.py', line 82, in collapse_excgroups",
                '    raise exc',
                "  File 'starlette/middleware/base.py', line 175, in __call__",
                '    response = await self.dispatch_func(request, call_next)',
                '               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^',
                "  File 'server/middleware.py', line 66, in __call__",
                '    self._check_tos(request)',
                "  File 'server/middleware.py', line 110, in _check_tos",
                '    decoded = jwt.decode(',
                '              ^^^^^^^^^^^',
                "  File 'jwt/api_jwt.py', line 222, in decode",
                '    decoded = self.decode_complete(',
                '              ^^^^^^^^^^^^^^^^^^^^^',
                "  File 'jwt/api_jwt.py', line 156, in decode_complete",
                '    decoded = api_jws.decode_complete(',
                '              ^^^^^^^^^^^^^^^^^^^^^^^^',
                "  File 'jwt/api_jws.py', line 220, in decode_complete",
                '    self._verify_signature(signing_input, header, signature, key, algorithms)',
                "  File 'jwt/api_jws.py', line 328, in _verify_signature",
                "    raise InvalidSignatureError('Signature verification failed')",
                "jwt.exceptions.InvalidSignatureError: Signature verification failed'",
            ]
            assert formatted == expected

    @freeze_time(FROZEN_TIMESTAMP)
    def test_filtering(self):
        # Ensure that secret values are still filtered
        string_io = StringIO()
        with (
            patch.dict(os.environ, {'my_secret_key': 'supersecretvalue'}),
            patch.object(openhands_logger.handlers[0], 'stream', string_io),
        ):
            openhands_logger.info('The secret key was supersecretvalue')
        output = json.loads(string_io.getvalue())
        assert output['message'] == 'The secret key was ******'
        assert output['severity'] == 'INFO'
        assert output['ts'] == FROZEN_TIMESTAMP
        assert 'module' in output
        assert 'funcName' in output
        assert 'lineno' in output

    @freeze_time(FROZEN_TIMESTAMP)
    def test_console_serializer_uses_ts_not_timestamp(self):
        """When LOG_JSON_FOR_CONSOLE=1, use 'ts' from custom_json_serializer, not 'timestamp'."""
        import server.logger as logger_module

        string_io = StringIO()
        logger = logging.Logger('test_console')

        # Patch LOG_JSON_FOR_CONSOLE to 1 for both setup_json_logger and custom_json_serializer
        with patch.object(logger_module, 'LOG_JSON_FOR_CONSOLE', 1):
            setup_json_logger(logger, 'INFO', _out=string_io)
            logger.info('Test console message')

        # Parse output - LOG_JSON_FOR_CONSOLE pretty-prints JSON across multiple lines
        output = json.loads(string_io.getvalue())

        # Should have 'ts' from custom_json_serializer but NOT 'timestamp'
        assert 'ts' in output
        assert 'timestamp' not in output
        assert output['message'] == 'Test console message'
        assert output['severity'] == 'INFO'

    @freeze_time(FROZEN_TIMESTAMP)
    def test_ts_not_duplicated_when_both_json_modes_enabled(self):
        """When both LOG_JSON=1 and LOG_JSON_FOR_CONSOLE=1, 'ts' should appear only once."""
        import server.logger as logger_module

        string_io = StringIO()
        logger = logging.Logger('test_both_modes')

        # Patch both LOG_JSON and LOG_JSON_FOR_CONSOLE to 1
        with (
            patch.object(logger_module, 'LOG_JSON', True),
            patch.object(logger_module, 'LOG_JSON_FOR_CONSOLE', 1),
        ):
            setup_json_logger(logger, 'INFO', _out=string_io)
            logger.info('Test both modes message')

        raw_output = string_io.getvalue()
        output = json.loads(raw_output)

        # Should have exactly one 'ts' field (not duplicated)
        assert 'ts' in output
        assert 'timestamp' not in output
        # Verify 'ts' appears only once in the raw output (not duplicated as key)
        assert (
            raw_output.count('"ts"') == 1
        ), f"'ts' should appear exactly once, found in: {raw_output}"
        assert output['message'] == 'Test both modes message'
        assert output['severity'] == 'INFO'
        # When LOG_JSON_FOR_CONSOLE=1, custom_json_serializer uses datetime.now().isoformat()
        # which doesn't include timezone info
        assert output['ts'] == FROZEN_TIMESTAMP_NO_TZ
