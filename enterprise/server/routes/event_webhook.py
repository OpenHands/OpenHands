import base64
import json
from enum import Enum
from typing import Annotated, Tuple

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)
from pydantic import BaseModel
from server.logger import logger
from server.utils.conversation_callback_utils import (
    process_event,
    update_agent_state,
    update_conversation_metadata,
    update_conversation_stats,
)
from storage.database import session_maker
from storage.stored_conversation_metadata_saas import StoredConversationMetadataSaas

from openhands.server.shared import conversation_manager

event_webhook_router = APIRouter(prefix='/event-webhook')


class BatchMethod(Enum):
    POST = 'POST'
    DELETE = 'DELETE'


class BatchOperation(BaseModel):
    method: BatchMethod
    path: str
    content: str | None = None
    encoding: str | None = None

    def get_content(self) -> bytes:
        if self.content is None:
            raise ValueError('empty_content_in_batch')
        if self.encoding == 'base64':
            return base64.b64decode(self.content.encode('ascii'))
        return self.content.encode('utf-8')

    def get_content_json(self) -> dict:
        return json.loads(self.get_content())


async def _process_batch_operations_background(
    batch_ops: list[BatchOperation],
    x_session_api_key: str | None,
):
    """Background task to process batched webhook requests with multiple file operations"""
    prev_conversation_id = None
    user_id = None

    for batch_op in batch_ops:
        try:
            if batch_op.method != BatchMethod.POST:
                # Log unhandled methods for future implementation
                logger.info(
                    'invalid_operation_in_batch_webhook',
                    extra={
                        'method': str(batch_op.method),
                        'path': batch_op.path,
                    },
                )
                continue

            # Updates to certain paths in the nested runtime are ignored
            if batch_op.path in {'settings.json', 'secrets.json'}:
                continue

            conversation_id, subpath = _parse_conversation_id_and_subpath(batch_op.path)

            # If the conversation id changes, then we must recheck the session_api_key
            if conversation_id != prev_conversation_id:
                user_id = _get_user_id(conversation_id)
                session_api_key = await _get_session_api_key(user_id, conversation_id)
                prev_conversation_id = conversation_id
                if session_api_key != x_session_api_key:
                    logger.error(
                        'authentication_failed_in_batch_webhook_background',
                        extra={
                            'conversation_id': conversation_id,
                            'user_id': user_id,
                            'path': batch_op.path,
                        },
                    )
                    continue  # Skip this operation but continue with others

            if subpath == 'agent_state.pkl':
                update_agent_state(user_id, conversation_id, batch_op.get_content())
                continue

            if subpath == 'conversation_stats.pkl':
                update_conversation_stats(
                    user_id, conversation_id, batch_op.get_content()
                )
                continue

            if subpath == 'metadata.json':
                update_conversation_metadata(
                    conversation_id, batch_op.get_content_json()
                )
                continue

            if subpath.startswith('events/'):
                await process_event(
                    user_id, conversation_id, subpath, batch_op.get_content_json()
                )
                continue

            if subpath.startswith('event_cache'):
                # No action required
                continue

            if subpath == 'exp_config.json':
                # No action required
                continue

            # Log unhandled paths for future implementation
            logger.warning(
                'unknown_path_in_batch_webhook',
                extra={
                    'path': subpath,
                    'user_id': user_id,
                    'conversation_id': conversation_id,
                },
            )
        except Exception as e:
            logger.error(
                f'error_processing_batch_operation: {type(e).__name__}: {e}',
                extra={
                    'path': batch_op.path,
                    'method': str(batch_op.method),
                },
                exc_info=True,
            )


@event_webhook_router.post('/batch')
async def on_batch_write(
    batch_ops: list[BatchOperation],
    background_tasks: BackgroundTasks,
    x_session_api_key: Annotated[str | None, Header()],
):
    """Handle batched webhook requests with multiple file operations in background"""
    # Add the batch processing to background tasks
    background_tasks.add_task(
        _process_batch_operations_background,
        batch_ops,
        x_session_api_key,
    )

    # Return immediately
    return Response(status_code=status.HTTP_202_ACCEPTED)


@event_webhook_router.post('/{path:path}')
async def on_write(
    path: str,
    request: Request,
    x_session_api_key: Annotated[str | None, Header()],
):
    """Handle writing conversation events and metadata"""
    conversation_id, subpath = _parse_conversation_id_and_subpath(path)
    user_id = _get_user_id(conversation_id)

    # Check the session API key to make sure this is from the correct conversation
    session_api_key = await _get_session_api_key(user_id, conversation_id)
    if session_api_key != x_session_api_key:
        return Response(status_code=status.HTTP_403_FORBIDDEN)

    if subpath == 'agent_state.pkl':
        content = await request.body()
        update_agent_state(user_id, conversation_id, content)
        return Response(status_code=status.HTTP_200_OK)

    try:
        content = await request.json()
    except Exception as exc:
        return Response(status_code=status.HTTP_400_BAD_REQUEST, content=str(exc))

    if subpath == 'metadata.json':
        update_conversation_metadata(conversation_id, content)
        return Response(status_code=status.HTTP_200_OK)

    if subpath.startswith('events/'):
        await process_event(user_id, conversation_id, subpath, content)
        return Response(status_code=status.HTTP_200_OK)

    if subpath.startswith('event_cache'):
        # No actionr required
        return Response(status_code=status.HTTP_200_OK)

    logger.error(
        'invalid_subpath_in_webhook',
        extra={
            'path': path,
            'user_id': user_id,
        },
    )
    return Response(status_code=status.HTTP_400_BAD_REQUEST)


@event_webhook_router.delete('/{path:path}')
async def on_delete(path: str, x_session_api_key: Annotated[str | None, Header()]):
    return Response(status_code=status.HTTP_200_OK)


def _parse_conversation_id_and_subpath(path: str) -> Tuple[str, str]:
    try:
        items = path.split('/')
        assert items[0] == 'sessions'
        conversation_id = items[1]
        subpath = '/'.join(items[2:])
        return conversation_id, subpath
    except (AssertionError, IndexError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST) from e


def _get_user_id(conversation_id: str) -> str:
    with session_maker() as session:
        conversation_metadata_saas = (
            session.query(StoredConversationMetadataSaas)
            .filter(StoredConversationMetadataSaas.conversation_id == conversation_id)
            .first()
        )
        return str(conversation_metadata_saas.user_id)


async def _get_session_api_key(user_id: str, conversation_id: str) -> str | None:
    agent_loop_info = await conversation_manager.get_agent_loop_info(
        user_id, filter_to_sids={conversation_id}
    )
    return agent_loop_info[0].session_api_key
