import json
import logging
from io import StringIO
from unittest.mock import patch

import pytest

from openhands.app_server.utils.logger import (
    LOG_JSON_LEVEL_KEY,
    OpenHandsLoggerAdapter,
    json_log_handler,
)
from openhands.app_server.utils.logger import openhands_logger as openhands_logger


@pytest.fixture
def test_handler():
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    openhands_logger.addHandler(handler)
    yield openhands_logger, stream
    openhands_logger.removeHandler(handler)


@pytest.fixture
def json_handler():
    stream = StringIO()
    json_handler = json_log_handler(logging.INFO, _out=stream)
    openhands_logger.addHandler(json_handler)
    yield openhands_logger, stream
    openhands_logger.removeHandler(json_handler)


def test_openai_api_key_masking(test_handler):
    logger, stream = test_handler

    api_key = 'sk-1234567890abcdef'
    message = f"OpenAI API key: api_key='{api_key}'and there's some stuff here"
    logger.info(message)
    log_output = stream.getvalue()
    assert api_key not in log_output


def test_azure_api_key_masking(test_handler):
    logger, stream = test_handler
    api_key = '1a2b3c4d5e6f7g8h9i0j'
    message = f"Azure API key: api_key='{api_key}' and chatty chat with ' and \" and '"
    logger.info(message)
    log_output = stream.getvalue()
    assert api_key not in log_output


def test_google_vertex_api_key_masking(test_handler):
    logger, stream = test_handler
    api_key = 'AIzaSyA1B2C3D4E5F6G7H8I9J0'
    message = f"Google Vertex API key: api_key='{api_key}' or not"
    logger.info(message)
    log_output = stream.getvalue()
    assert api_key not in log_output


def test_anthropic_api_key_masking(test_handler):
    logger, stream = test_handler
    api_key = 'sk-ant-1234567890abcdef-some-more-stuff-here'
    message = f"Anthropic API key: api_key='{api_key}' and there's some 'stuff' here"
    logger.info(message)
    log_output = stream.getvalue()
    assert api_key not in log_output


def test_app_config_attributes_masking(test_handler):
    logger, stream = test_handler
    # Test that sensitive config-like strings are masked in logs
    config_str = "AppConfig(search_api_key='search-xyz789', github_token='ghp_abcdefghijklmnopqrstuvwxyz')"
    logger.info(f'App Config: {config_str}')
    log_output = stream.getvalue()
    assert 'search-xyz789' not in log_output
    assert 'ghp_abcdefghijklmnopqrstuvwxyz' not in log_output


def test_sensitive_env_vars_masking(test_handler):
    logger, stream = test_handler
    environ = {
        'API_KEY': 'API_KEY_VALUE',
        'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID_VALUE',
        'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY_VALUE',
        'E2B_API_KEY': 'E2B_API_KEY_VALUE',
        'GITHUB_TOKEN': 'GITHUB_TOKEN_VALUE',
        'JWT_SECRET': 'JWT_SECRET_VALUE',
    }

    with patch.dict(
        'openhands.app_server.utils.logger.os.environ', environ, clear=True
    ):
        log_message = ' '.join(f"{attr}='{value}'" for attr, value in environ.items())
        logger.info(log_message)

        log_output = stream.getvalue()
        for _, value in environ.items():
            assert value not in log_output


def test_special_cases_masking(test_handler):
    logger, stream = test_handler
    environ = {
        'LLM_API_KEY': 'LLM_API_KEY_VALUE',
        'SANDBOX_ENV_GITHUB_TOKEN': 'SANDBOX_ENV_GITHUB_TOKEN_VALUE',
    }

    with patch.dict(
        'openhands.app_server.utils.logger.os.environ', environ, clear=True
    ):
        log_message = ' '.join(
            f"{attr}={value} with no single quotes' and something"
            for attr, value in environ.items()
        )
        logger.info(log_message)

        log_output = stream.getvalue()
        for attr, value in environ.items():
            assert value not in log_output


class TestJsonOutput:
    def test_info(self, json_handler):
        logger, string_io = json_handler

        logger.info('Test message')
        output = json.loads(string_io.getvalue())
        assert 'timestamp' in output
        del output['timestamp']
        assert output == {'message': 'Test message', LOG_JSON_LEVEL_KEY: 'INFO'}

    def test_error(self, json_handler):
        logger, string_io = json_handler

        logger.error('Test message')
        output = json.loads(string_io.getvalue())
        del output['timestamp']
        assert output == {'message': 'Test message', LOG_JSON_LEVEL_KEY: 'ERROR'}

    def test_extra_fields(self, json_handler):
        logger, string_io = json_handler

        logger.info('Test message', extra={'key': '..val..'})
        output = json.loads(string_io.getvalue())
        del output['timestamp']
        assert output == {
            'key': '..val..',
            'message': 'Test message',
            LOG_JSON_LEVEL_KEY: 'INFO',
        }

    def test_extra_fields_from_adapter(self, json_handler):
        logger, string_io = json_handler
        subject = OpenHandsLoggerAdapter(logger, extra={'context_field': '..val..'})
        subject.info('Test message', extra={'log_fied': '..val..'})
        output = json.loads(string_io.getvalue())
        del output['timestamp']
        assert output == {
            'context_field': '..val..',
            'log_fied': '..val..',
            'message': 'Test message',
            LOG_JSON_LEVEL_KEY: 'INFO',
        }

    def test_extra_fields_from_adapter_can_override(self, json_handler):
        logger, string_io = json_handler
        subject = OpenHandsLoggerAdapter(logger, extra={'override': 'a'})
        subject.info('Test message', extra={'override': 'b'})
        output = json.loads(string_io.getvalue())
        del output['timestamp']
        assert output == {
            'override': 'b',
            'message': 'Test message',
            LOG_JSON_LEVEL_KEY: 'INFO',
        }
