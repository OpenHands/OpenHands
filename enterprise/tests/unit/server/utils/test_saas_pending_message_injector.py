from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from openhands.app_server.errors import AuthError
from server.utils.saas_pending_message_injector import SaasSQLPendingMessageService


@pytest.mark.parametrize(
    "conversation_alias",
    [
        lambda conversation_id: conversation_id.hex,
        lambda conversation_id: str(conversation_id).upper(),
        lambda conversation_id: conversation_id.hex.upper(),
    ],
)
@pytest.mark.asyncio
async def test_ownership_check_normalizes_conversation_uuid(conversation_alias):
    conversation_id = uuid4()
    user_id = uuid4()
    owner_id = uuid4()
    result = MagicMock()
    result.scalar_one_or_none.return_value = SimpleNamespace(user_id=owner_id)
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    user_context = MagicMock()
    user_context.get_user_id = AsyncMock(return_value=str(user_id))
    service = SaasSQLPendingMessageService(session, user_context)

    with pytest.raises(AuthError):
        await service._validate_conversation_ownership(
            conversation_alias(conversation_id)
        )

    statement = session.execute.await_args.args[0]
    assert str(conversation_id) in statement.compile().params.values()


@pytest.mark.asyncio
async def test_task_alias_enforces_effective_organization():
    task_id = uuid4()
    conversation_id = uuid4()
    user_id = uuid4()
    task_result = MagicMock()
    task_result.scalar_one_or_none.return_value = conversation_id
    metadata_result = MagicMock()
    metadata_result.scalar_one_or_none.return_value = SimpleNamespace(
        user_id=user_id,
        org_id=uuid4(),
    )
    session = MagicMock()
    session.execute = AsyncMock(side_effect=[task_result, metadata_result])
    user_context = MagicMock()
    user_context.get_user_id = AsyncMock(return_value=str(user_id))
    user_context.user_auth.get_effective_org_id = AsyncMock(return_value=uuid4())
    service = SaasSQLPendingMessageService(session, user_context)

    with pytest.raises(AuthError, match="different organization"):
        await service._validate_conversation_ownership(f"task-{task_id.hex}")


@pytest.mark.asyncio
async def test_task_alias_accepts_matching_organization():
    task_id = uuid4()
    conversation_id = uuid4()
    user_id = uuid4()
    organization_id = uuid4()
    task_result = MagicMock()
    task_result.scalar_one_or_none.return_value = conversation_id
    metadata_result = MagicMock()
    metadata_result.scalar_one_or_none.return_value = SimpleNamespace(
        user_id=user_id,
        org_id=organization_id,
    )
    session = MagicMock()
    session.execute = AsyncMock(side_effect=[task_result, metadata_result])
    user_context = MagicMock()
    user_context.get_user_id = AsyncMock(return_value=str(user_id))
    user_context.user_auth.get_effective_org_id = AsyncMock(
        return_value=organization_id
    )
    service = SaasSQLPendingMessageService(session, user_context)

    await service._validate_conversation_ownership(f"task-{task_id.hex}")


@pytest.mark.asyncio
async def test_unmapped_task_alias_is_denied():
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    user_context = MagicMock()
    user_context.get_user_id = AsyncMock(return_value=str(uuid4()))
    service = SaasSQLPendingMessageService(session, user_context)

    with pytest.raises(AuthError, match="not available"):
        await service._validate_conversation_ownership(f"task-{uuid4().hex}")
