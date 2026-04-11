import logging
from unittest.mock import patch

from openhands.core.logger import RedactURLParamsFilter, SensitiveDataFilter


@patch.dict(
    'os.environ',
    {
        'API_SECRET': 'super-secret-123',
        'AUTH_TOKEN': 'auth-token-456',
        'NORMAL_VAR': 'normal-value',
    },
    clear=True,
)
def test_sensitive_data_filter_basic():
    # Create a filter instance
    filter = SensitiveDataFilter()

    # Create a log record with sensitive data
    record = logging.LogRecord(
        name='test_logger',
        level=logging.INFO,
        pathname='test.py',
        lineno=1,
        msg='API Secret: super-secret-123, Token: auth-token-456, Normal: normal-value',
        args=(),
        exc_info=None,
    )

    # Apply the filter
    filter.filter(record)

    # Check that sensitive data is masked but normal data isn't
    assert '******' in record.msg
    assert 'super-secret-123' not in record.msg
    assert 'auth-token-456' not in record.msg
    assert 'normal-value' in record.msg


@patch.dict('os.environ', {}, clear=True)
def test_sensitive_data_filter_empty_values():
    # Test with empty environment variables
    filter = SensitiveDataFilter()

    record = logging.LogRecord(
        name='test_logger',
        level=logging.INFO,
        pathname='test.py',
        lineno=1,
        msg='No sensitive data here',
        args=(),
        exc_info=None,
    )

    # Apply the filter
    filter.filter(record)

    # Message should remain unchanged
    assert record.msg == 'No sensitive data here'


@patch.dict('os.environ', {'API_KEY': 'secret-key-789'}, clear=True)
def test_sensitive_data_filter_multiple_occurrences():
    # Test with multiple occurrences of the same sensitive data
    filter = SensitiveDataFilter()

    # Create a message with multiple occurrences of the same sensitive data
    record = logging.LogRecord(
        name='test_logger',
        level=logging.INFO,
        pathname='test.py',
        lineno=1,
        msg='Key1: secret-key-789, Key2: secret-key-789',
        args=(),
        exc_info=None,
    )

    # Apply the filter
    filter.filter(record)

    # Check that all occurrences are masked
    assert record.msg.count('******') == 2
    assert 'secret-key-789' not in record.msg


@patch.dict(
    'os.environ',
    {
        'secret_KEY': 'secret-value-1',
        'API_secret': 'secret-value-2',
        'TOKEN_code': 'secret-value-3',
    },
    clear=True,
)
def test_sensitive_data_filter_case_sensitivity():
    # Test with different case variations in environment variable names
    filter = SensitiveDataFilter()

    record = logging.LogRecord(
        name='test_logger',
        level=logging.INFO,
        pathname='test.py',
        lineno=1,
        msg='Values: secret-value-1, secret-value-2, secret-value-3',
        args=(),
        exc_info=None,
    )

    # Apply the filter
    filter.filter(record)

    # Check that all sensitive values are masked regardless of case
    assert 'secret-value-1' not in record.msg
    assert 'secret-value-2' not in record.msg
    assert 'secret-value-3' not in record.msg
    assert record.msg.count('******') == 3


# --------------------------------------------------------------------------
# RedactURLParamsFilter tests
# --------------------------------------------------------------------------


def test_redact_url_params_filter_websocket_log():
    """Test that session_api_key is redacted from WebSocket access logs."""
    log_filter = RedactURLParamsFilter()

    # Simulate uvicorn WebSocket access log format
    record = logging.LogRecord(
        name='uvicorn.access',
        level=logging.INFO,
        pathname='',
        lineno=0,
        msg='%s - "%s" [%s]',
        args=(
            '127.0.0.1:8000',
            'GET /ws/abc123?resend_all=true&session_api_key=secret-token-12345',
            'accepted',
        ),
        exc_info=None,
    )

    # Apply the filter
    result = log_filter.filter(record)

    # Filter should always return True (never drop records)
    assert result is True

    # Check that secret is redacted but other params preserved
    args_str = str(record.args)
    assert 'secret-token-12345' not in args_str
    # URL-encoded <redacted> is %3Credacted%3E
    assert '<redacted>' in args_str or '%3Credacted%3E' in args_str
    assert 'resend_all=true' in args_str


def test_redact_url_params_filter_multiple_sensitive_params():
    """Test that multiple sensitive parameters are redacted."""
    log_filter = RedactURLParamsFilter()

    record = logging.LogRecord(
        name='uvicorn.access',
        level=logging.INFO,
        pathname='',
        lineno=0,
        msg='Request: %s',
        args=('GET /api?api_key=secret1&token=secret2&user_id=123',),
        exc_info=None,
    )

    log_filter.filter(record)

    args_str = str(record.args)
    assert 'secret1' not in args_str
    assert 'secret2' not in args_str
    assert 'user_id=123' in args_str


def test_redact_url_params_filter_non_url_passthrough():
    """Test that messages without URLs pass through unchanged."""
    log_filter = RedactURLParamsFilter()

    record = logging.LogRecord(
        name='test',
        level=logging.INFO,
        pathname='',
        lineno=0,
        msg='Normal log: %s %s',
        args=('hello', 'world'),
        exc_info=None,
    )

    log_filter.filter(record)

    # Message should remain unchanged
    assert record.args == ('hello', 'world')


def test_redact_url_params_filter_no_query_string():
    """Test that URLs without query strings pass through unchanged."""
    log_filter = RedactURLParamsFilter()

    record = logging.LogRecord(
        name='test',
        level=logging.INFO,
        pathname='',
        lineno=0,
        msg='Request: %s',
        args=('GET /api/v1/users',),
        exc_info=None,
    )

    log_filter.filter(record)

    # URL without query string should remain unchanged
    assert record.args == ('GET /api/v1/users',)


def test_redact_url_params_filter_empty_args():
    """Test that records with no args are handled gracefully."""
    log_filter = RedactURLParamsFilter()

    record = logging.LogRecord(
        name='test',
        level=logging.INFO,
        pathname='',
        lineno=0,
        msg='Simple message',
        args=(),
        exc_info=None,
    )

    result = log_filter.filter(record)

    assert result is True
    assert record.args == ()


def test_redact_url_params_filter_none_args():
    """Test that records with None args are handled gracefully."""
    log_filter = RedactURLParamsFilter()

    record = logging.LogRecord(
        name='test',
        level=logging.INFO,
        pathname='',
        lineno=0,
        msg='Simple message',
        args=None,
        exc_info=None,
    )

    result = log_filter.filter(record)

    assert result is True
    assert record.args is None


def test_redact_url_params_filter_dict_args():
    """Test that records with dict args (used by some formatters) pass through."""
    log_filter = RedactURLParamsFilter()

    record = logging.LogRecord(
        name='test',
        level=logging.INFO,
        pathname='',
        lineno=0,
        msg='%(method)s %(path)s',
        args={'method': 'GET', 'path': '/api?secret=test'},
        exc_info=None,
    )

    result = log_filter.filter(record)

    # Dict args should pass through (filter only handles tuple/list)
    assert result is True
    assert record.args == {'method': 'GET', 'path': '/api?secret=test'}
