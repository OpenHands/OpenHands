from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from openhands.app_server.recaptcha.recaptcha_models import RecaptchaLog
from openhands.app_server.recaptcha.sql_recaptcha_service import RecaptchaService


@pytest.fixture
def mock_db_session():
    return AsyncMock()


@pytest.fixture
def mock_user_context():
    context = AsyncMock()
    context.get_user_id = AsyncMock(return_value='user-123')
    return context


@pytest.mark.asyncio
async def test_save_log(mock_db_session, mock_user_context):
    service = RecaptchaService(
        db_session=mock_db_session, user_context=mock_user_context
    )

    log_id = uuid4()
    log = RecaptchaLog(
        id=log_id,
        score=0.9,
        token='some-token',
        action='login',
        hostname='example.com',
        apk_package_name=None,
    )

    result = await service.save_log(log)

    assert result == log
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()

    # Verify the stored object
    stored = mock_db_session.add.call_args[0][0]
    assert stored.id == str(log_id)
    assert stored.user_id == 'user-123'
    assert stored.score == 0.9
    assert stored.token == 'some-token'
    assert stored.action == 'login'
    assert stored.hostname == 'example.com'
    assert stored.apk_package_name is None
