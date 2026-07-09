from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from openhands.app_server.config import get_db_session, get_user_context
from openhands.app_server.recaptcha.recaptcha_models import (
    RecaptchaLog,
    RecaptchaLogRequest,
)
from openhands.app_server.recaptcha.sql_recaptcha_service import RecaptchaService
from openhands.app_server.user.user_context import UserContext

router = APIRouter(prefix='/recaptcha', tags=['recaptcha'])


@router.post('/log', status_code=status.HTTP_201_CREATED)
async def log_recaptcha(
    request: RecaptchaLogRequest,
    user_context: UserContext = Depends(get_user_context),
    db_session: AsyncSession = Depends(get_db_session),
):
    """Log reCAPTCHA verification result."""
    service = RecaptchaService(db_session=db_session, user_context=user_context)
    log = RecaptchaLog(
        score=request.score,
        token=request.token,
        action=request.action,
        hostname=request.hostname,
        apk_package_name=request.apk_package_name,
    )
    await service.save_log(log)
    return {'status': 'success'}
