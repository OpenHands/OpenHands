"""Shared Conversation router for OpenHands Server."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from server.sharing.shared_conversation_info_service import (
    SharedConversationInfoService,
)
from server.sharing.shared_conversation_models import (
    SharedConversation,
)
from server.sharing.sql_shared_conversation_info_service import (
    SQLSharedConversationInfoServiceInjector,
)

router = APIRouter(prefix='/api/shared-conversations', tags=['Sharing'])
shared_conversation_info_service_dependency = Depends(
    SQLSharedConversationInfoServiceInjector().depends
)


# Read methods
#
# These endpoints are unauthenticated. Only batch lookup by known IDs is
# exposed publicly so that share links of the form
# /shared/conversations/<id> can be viewed without auth. Listing or
# enumerating shared conversations is intentionally not exposed.


@router.get('')
async def batch_get_shared_conversations(
    ids: Annotated[list[str], Query()],
    shared_conversation_service: SharedConversationInfoService = shared_conversation_info_service_dependency,
) -> list[SharedConversation | None]:
    """Get a batch of shared conversations given their ids. Return None for any missing or non-shared."""
    if len(ids) > 100:
        raise HTTPException(
            status_code=400,
            detail=f'Cannot request more than 100 conversations at once, got {len(ids)}',
        )
    uuids = [UUID(id_) for id_ in ids]
    shared_conversation_info = (
        await shared_conversation_service.batch_get_shared_conversation_info(uuids)
    )
    return shared_conversation_info
