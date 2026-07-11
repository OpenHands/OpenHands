from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    String,
)
from sqlalchemy.ext.asyncio import AsyncSession

from openhands.agent_server.utils import utc_now
from openhands.app_server.recaptcha.recaptcha_models import RecaptchaLog
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.utils.sql_utils import Base

logger = logging.getLogger(__name__)


class RecaptchaStore(Base):  # type: ignore
    __tablename__ = 'recaptcha_logs'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=True)
    score = Column(Float, nullable=True)
    token = Column(String, nullable=True)
    action = Column(String, nullable=True)
    hostname = Column(String, nullable=True)
    apk_package_name = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)


@dataclass
class RecaptchaService:
    """Service for logging reCAPTCHA verification results."""

    db_session: AsyncSession
    user_context: UserContext

    async def save_log(self, log: RecaptchaLog) -> RecaptchaLog:
        """Save a reCAPTCHA log entry to the database."""

        user_id = await self.user_context.get_user_id()

        stored = RecaptchaStore(
            id=str(log.id),
            user_id=user_id,
            score=log.score,
            token=log.token,
            action=log.action,
            hostname=log.hostname,
            apk_package_name=log.apk_package_name,
            created_at=log.created_at,
        )

        self.db_session.add(stored)
        await self.db_session.commit()
        return log
