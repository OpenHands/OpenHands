from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from openhands.app_server.app_conversation.app_conversation_models import (
    CODEX_CREDENTIAL_BINDING_TAG_KEY,
    CODEX_CREDENTIAL_BINDING_TAG_VALUE,
    AppConversationInfo,
    AppConversationInfoPage,
)
from openhands.app_server.sandbox.sandbox_router import resume_sandbox


@pytest.mark.asyncio
async def test_raw_resume_rejects_managed_conversation():
    conversation_service = AsyncMock()
    conversation_service.search_app_conversation_info.return_value = (
        AppConversationInfoPage(
            items=[
                AppConversationInfo(
                    created_by_user_id='user',
                    sandbox_id='sandbox',
                    tags={
                        CODEX_CREDENTIAL_BINDING_TAG_KEY: (
                            CODEX_CREDENTIAL_BINDING_TAG_VALUE
                        )
                    },
                )
            ]
        )
    )
    sandbox_service = AsyncMock()

    with pytest.raises(HTTPException) as exc_info:
        await resume_sandbox(
            'sandbox',
            sandbox_service=sandbox_service,
            app_conversation_info_service=conversation_service,
        )

    assert exc_info.value.status_code == 409
    sandbox_service.resume_sandbox.assert_not_awaited()


@pytest.mark.asyncio
async def test_raw_resume_allows_unmanaged_sandbox():
    conversation_service = AsyncMock()
    conversation_service.search_app_conversation_info.return_value = (
        AppConversationInfoPage(items=[])
    )
    sandbox_service = AsyncMock()
    sandbox_service.resume_sandbox.return_value = True

    await resume_sandbox(
        'sandbox',
        sandbox_service=sandbox_service,
        app_conversation_info_service=conversation_service,
    )

    sandbox_service.resume_sandbox.assert_awaited_once_with('sandbox')


@pytest.mark.asyncio
async def test_raw_resume_checks_every_conversation_page():
    conversation_service = AsyncMock()
    conversation_service.search_app_conversation_info.side_effect = [
        AppConversationInfoPage(items=[], next_page_id='next'),
        AppConversationInfoPage(
            items=[
                AppConversationInfo(
                    created_by_user_id='user',
                    sandbox_id='sandbox',
                    tags={
                        CODEX_CREDENTIAL_BINDING_TAG_KEY: (
                            CODEX_CREDENTIAL_BINDING_TAG_VALUE
                        )
                    },
                )
            ]
        ),
    ]
    sandbox_service = AsyncMock()

    with pytest.raises(HTTPException):
        await resume_sandbox(
            'sandbox',
            sandbox_service=sandbox_service,
            app_conversation_info_service=conversation_service,
        )

    assert conversation_service.search_app_conversation_info.await_count == 2
    sandbox_service.resume_sandbox.assert_not_awaited()
