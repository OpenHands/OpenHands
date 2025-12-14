"""Public API routes for conversation sharing.

These endpoints provide read-only access to public conversations without authentication.
They are designed with security-first principles to prevent data leakage.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status

from openhands.core.logger import openhands_logger as logger
from openhands.server.data_models.public_conversation import (
    PublicConversationDetail,
    PublicConversationInfo,
    PublicMessageInfo,
)
from openhands.server.services.public_conversation_service import (
    PublicConversationService,
)
from openhands.server.utils import get_conversation_store
from openhands.storage.conversation.conversation_store import ConversationStore

# Create router with public prefix for easy security auditing
app = APIRouter(prefix='/api/public/conversations')


async def get_public_conversation_service(
    conversation_store: ConversationStore = Depends(get_conversation_store),
) -> PublicConversationService:
    """Dependency to get public conversation service."""
    return PublicConversationService(conversation_store)


@app.get('/{conversation_id}', response_model=PublicConversationInfo)
async def get_public_conversation(
    conversation_id: str,
    request: Request,
    service: PublicConversationService = Depends(get_public_conversation_service),
) -> PublicConversationInfo:
    """Get public conversation metadata.

    Args:
        conversation_id: The conversation ID
        request: FastAPI request object for logging
        service: Public conversation service

    Returns:
        Public conversation information

    Raises:
        HTTPException: 404 if conversation not found or not public
    """
    # Log public access attempt for security monitoring
    client_ip = request.client.host if request.client else 'unknown'
    logger.info(
        f'Public conversation access attempt: {conversation_id} from {client_ip}',
        extra={
            'conversation_id': conversation_id,
            'client_ip': client_ip,
            'endpoint': 'get_public_conversation',
        },
    )

    conversation = await service.get_public_conversation(conversation_id)
    if not conversation:
        logger.warning(
            f'Public conversation not found or not public: {conversation_id}',
            extra={'conversation_id': conversation_id, 'client_ip': client_ip},
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Conversation not found or not public',
        )

    logger.info(
        f'Public conversation served: {conversation_id}',
        extra={'conversation_id': conversation_id, 'client_ip': client_ip},
    )

    return conversation


@app.get('/{conversation_id}/messages', response_model=list[PublicMessageInfo])
async def get_public_conversation_messages(
    conversation_id: str,
    request: Request,
    service: PublicConversationService = Depends(get_public_conversation_service),
) -> list[PublicMessageInfo]:
    """Get public conversation messages.

    Args:
        conversation_id: The conversation ID
        request: FastAPI request object for logging
        service: Public conversation service

    Returns:
        List of public-safe messages

    Raises:
        HTTPException: 404 if conversation not found or not public
    """
    # Log public access attempt for security monitoring
    client_ip = request.client.host if request.client else 'unknown'
    logger.info(
        f'Public conversation messages access: {conversation_id} from {client_ip}',
        extra={
            'conversation_id': conversation_id,
            'client_ip': client_ip,
            'endpoint': 'get_public_conversation_messages',
        },
    )

    # First verify conversation is public
    conversation = await service.get_public_conversation(conversation_id)
    if not conversation:
        logger.warning(
            f'Public conversation messages not found or not public: {conversation_id}',
            extra={'conversation_id': conversation_id, 'client_ip': client_ip},
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Conversation not found or not public',
        )

    messages = await service.get_public_conversation_messages(conversation_id)

    logger.info(
        f'Public conversation messages served: {conversation_id} ({len(messages)} messages)',
        extra={
            'conversation_id': conversation_id,
            'client_ip': client_ip,
            'message_count': len(messages),
        },
    )

    return messages


@app.get('/{conversation_id}/full', response_model=PublicConversationDetail)
async def get_public_conversation_full(
    conversation_id: str,
    request: Request,
    service: PublicConversationService = Depends(get_public_conversation_service),
) -> PublicConversationDetail:
    """Get complete public conversation with messages.

    Args:
        conversation_id: The conversation ID
        request: FastAPI request object for logging
        service: Public conversation service

    Returns:
        Complete public conversation with messages

    Raises:
        HTTPException: 404 if conversation not found or not public
    """
    # Log public access attempt for security monitoring
    client_ip = request.client.host if request.client else 'unknown'
    logger.info(
        f'Public conversation full access: {conversation_id} from {client_ip}',
        extra={
            'conversation_id': conversation_id,
            'client_ip': client_ip,
            'endpoint': 'get_public_conversation_full',
        },
    )

    conversation_detail = await service.get_public_conversation_detail(conversation_id)
    if not conversation_detail:
        logger.warning(
            f'Public conversation full not found or not public: {conversation_id}',
            extra={'conversation_id': conversation_id, 'client_ip': client_ip},
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Conversation not found or not public',
        )

    logger.info(
        f'Public conversation full served: {conversation_id} ({len(conversation_detail.messages)} messages)',
        extra={
            'conversation_id': conversation_id,
            'client_ip': client_ip,
            'message_count': len(conversation_detail.messages),
        },
    )

    return conversation_detail


@app.get('/token/{share_token}', response_model=PublicConversationInfo)
async def get_public_conversation_by_token(
    share_token: str,
    request: Request,
    service: PublicConversationService = Depends(get_public_conversation_service),
) -> PublicConversationInfo:
    """Get public conversation by share token.

    Args:
        share_token: The public share token
        request: FastAPI request object for logging
        service: Public conversation service

    Returns:
        Public conversation information

    Raises:
        HTTPException: 404 if token not found or invalid
    """
    # Log public access attempt for security monitoring
    client_ip = request.client.host if request.client else 'unknown'
    logger.info(
        f'Public conversation token access: {share_token[:8]}... from {client_ip}',
        extra={
            'share_token_prefix': share_token[:8],
            'client_ip': client_ip,
            'endpoint': 'get_public_conversation_by_token',
        },
    )

    conversation = await service.get_public_conversation_by_token(share_token)
    if not conversation:
        logger.warning(
            f'Public conversation token not found: {share_token[:8]}...',
            extra={'share_token_prefix': share_token[:8], 'client_ip': client_ip},
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Invalid or expired share token',
        )

    logger.info(
        f'Public conversation served via token: {conversation.conversation_id}',
        extra={
            'conversation_id': conversation.conversation_id,
            'client_ip': client_ip,
            'share_token_prefix': share_token[:8],
        },
    )

    return conversation


# Health check endpoint for public API
@app.get('/health')
async def public_api_health() -> dict[str, str]:
    """Health check for public conversation API."""
    return {'status': 'healthy', 'service': 'public_conversations'}
