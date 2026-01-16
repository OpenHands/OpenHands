from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.future import select
from storage.database import session_maker
from storage.feedback import ConversationFeedback
from storage.stored_conversation_metadata_saas import StoredConversationMetadataSaas

from openhands.events.event_store import EventStore
from openhands.server.shared import file_store
from openhands.server.user_auth import get_user_id
from openhands.utils.async_utils import call_sync_from_async

router = APIRouter(prefix='/feedback', tags=['feedback'])


async def get_event_ids(conversation_id: str, user_id: str) -> List[int]:
    """Get all event IDs for a given conversation.

    Args:
        conversation_id: The ID of the conversation to get events for
        user_id: The ID of the user who owns the conversation

    Returns:
        List of event IDs in the conversation

    Raises:
        HTTPException: If conversation metadata not found
    """

    # Verify the conversation belongs to the user
    def _verify_conversation():
        with session_maker() as session:
            metadata = (
                session.query(StoredConversationMetadataSaas)
                .filter(
                    StoredConversationMetadataSaas.conversation_id == conversation_id,
                    StoredConversationMetadataSaas.user_id == user_id,
                )
                .first()
            )
            if not metadata:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f'Conversation {conversation_id} not found',
                )

    await call_sync_from_async(_verify_conversation)

    # Create an event store to access the events directly
    # This works even when the conversation is not running
    event_store = EventStore(
        sid=conversation_id,
        file_store=file_store,
        user_id=user_id,
    )

    # Get events from the event store
    events = event_store.search_events(start_id=0)

    # Return list of event IDs
    return [event.id for event in events]


class FeedbackRequest(BaseModel):
    conversation_id: str
    event_id: Optional[int] = None
    rating: int = Field(..., ge=1, le=5)
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@router.post('/conversation', status_code=status.HTTP_201_CREATED)
async def submit_conversation_feedback(feedback: FeedbackRequest):
    """
    Submit feedback for a conversation.

    This endpoint accepts a rating (1-5) and optional reason for the feedback.
    The feedback is associated with a specific conversation and optionally a specific event.
    """
    # Validate rating is between 1 and 5
    if feedback.rating < 1 or feedback.rating > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Rating must be between 1 and 5',
        )

    # Create new feedback record
    new_feedback = ConversationFeedback(
        conversation_id=feedback.conversation_id,
        event_id=feedback.event_id,
        rating=feedback.rating,
        reason=feedback.reason,
        metadata=feedback.metadata,
    )

    # Add to database
    def _save_feedback():
        with session_maker() as session:
            session.add(new_feedback)
            session.commit()

    await call_sync_from_async(_save_feedback)

    return {'status': 'success', 'message': 'Feedback submitted successfully'}


@router.get('/conversation/{conversation_id}/batch')
async def get_batch_feedback(conversation_id: str, user_id: str = Depends(get_user_id)):
    """
    Get feedback for all events in a conversation.

    Returns feedback status for each event, including whether feedback exists
    and if so, the rating and reason.
    """
    # Get all event IDs for the conversation
    event_ids = await get_event_ids(conversation_id, user_id)
    if not event_ids:
        return {}

    # Query for existing feedback for all events
    def _check_feedback():
        with session_maker() as session:
            result = session.execute(
                select(ConversationFeedback).where(
                    ConversationFeedback.conversation_id == conversation_id,
                    ConversationFeedback.event_id.in_(event_ids),
                )
            )

            # Create a mapping of event_id to feedback
            feedback_map = {
                feedback.event_id: {
                    'exists': True,
                    'rating': feedback.rating,
                    'reason': feedback.reason,
                }
                for feedback in result.scalars()
            }

            # Build response including all events
            response = {}
            for event_id in event_ids:
                response[str(event_id)] = feedback_map.get(event_id, {'exists': False})

            return response

    return await call_sync_from_async(_check_feedback)
