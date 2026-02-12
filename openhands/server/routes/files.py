# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
import os
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from openhands.core.exceptions import AgentRuntimeUnavailableError
from openhands.core.logger import openhands_logger as logger
from openhands.runtime.base import Runtime
from openhands.server.dependencies import get_dependencies
from openhands.server.file_config import FILES_TO_IGNORE
from openhands.server.files import POSTUploadFilesModel
from openhands.server.session.conversation import ServerConversation
from openhands.server.shared import conversation_manager
from openhands.server.user_auth import get_user_id
from openhands.server.utils import (
    get_conversation,
    get_conversation_metadata,
    get_conversation_store,
)
from openhands.storage.conversation.conversation_store import ConversationStore
from openhands.storage.data_models.conversation_metadata import ConversationMetadata
from openhands.utils.async_utils import call_sync_from_async

app = APIRouter(
    prefix='/api/conversations/{conversation_id}', dependencies=get_dependencies()
)


@app.get(
    '/list-files',
    response_model=list[str],
    responses={
        404: {'description': 'Runtime not initialized', 'model': dict},
        500: {'description': 'Error listing or filtering files', 'model': dict},
    },
    deprecated=True,
)
async def list_files(
    metadata: ConversationMetadata = Depends(get_conversation_metadata),
    path: str | None = None,
) -> list[str] | JSONResponse:
    """List files in the specified path.

    This function retrieves a list of files from the agent's runtime file store,
    excluding certain system and hidden files/directories.

    To list files:
    ```sh
    curl http://localhost:3000/api/conversations/{conversation_id}/list-files
    ```

    Args:
        metadata: The conversation metadata (provides conversation_id and user access validation).
        path (str, optional): The path to list files from. Defaults to None.

    Returns:
        list: A list of file names in the specified path.

    Raises:
        HTTPException: If there's an error listing the files.

        For V1 conversations, file operations are handled through the agent server.
        Use the sandbox's exposed agent server URL to access file operations.
    """
    conversation_id = metadata.conversation_id
    try:
        file_list = await conversation_manager.list_files(conversation_id, path)
    except ValueError as e:
        logger.error(f'Error listing files: {e}')
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={'error': str(e)},
        )
    except AgentRuntimeUnavailableError as e:
        logger.error(f'Error listing files: {e}')
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': f'Error listing files: {e}'},
        )
    except httpx.TimeoutException:
        logger.error(f'Timeout listing files for conversation {conversation_id}')
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={'error': 'Request to runtime timed out'},
        )
    except httpx.ConnectError:
        logger.error(
            f'Connection error listing files for conversation {conversation_id}'
        )
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={'error': 'Unable to connect to runtime'},
        )
    except httpx.HTTPStatusError as e:
        logger.error(f'HTTP error listing files: {e.response.status_code}')
        return JSONResponse(
            status_code=e.response.status_code,
            content={'error': f'Runtime returned error: {e.response.status_code}'},
        )
    except Exception as e:
        logger.error(f'Error listing files: {e}')
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': f'Error listing files: {e}'},
        )

    file_list = [f for f in file_list if f not in FILES_TO_IGNORE]

    return file_list


# NOTE: We use response_model=None for endpoints that can return multiple response types
# (like FileResponse | JSONResponse). This is because FastAPI's response_model expects a
# Pydantic model, but Starlette response classes like FileResponse are not Pydantic models.
# Instead, we document the possible responses using the 'responses' parameter and maintain
# proper type annotations for mypy.
@app.get(
    '/select-file',
    response_model=None,
    responses={
        200: {'description': 'File content returned as JSON', 'model': dict[str, str]},
        500: {'description': 'Error opening file', 'model': dict},
        415: {'description': 'Unsupported media type', 'model': dict},
    },
    deprecated=True,
)
async def select_file(
    file: str,
    metadata: ConversationMetadata = Depends(get_conversation_metadata),
) -> FileResponse | JSONResponse:
    """Retrieve the content of a specified file.

    To select a file:
    ```sh
    curl http://localhost:3000/api/conversations/{conversation_id}/select-file?file=<file_path>
    ```

    Args:
        file (str): The path of the file to be retrieved (relative to workspace root).
        metadata: The conversation metadata (provides conversation_id and user access validation).

    Returns:
        dict: A dictionary containing the file content.

    Raises:
        HTTPException: If there's an error opening the file.

        For V1 conversations, file operations are handled through the agent server.
        Use the sandbox's exposed agent server URL to access file operations.
    """
    conversation_id = metadata.conversation_id
    try:
        content, error = await conversation_manager.select_file(conversation_id, file)
    except ValueError as e:
        logger.error(f'Error opening file {file}: {e}')
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={'error': str(e)},
        )
    except AgentRuntimeUnavailableError as e:
        logger.error(f'Error opening file {file}: {e}')
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': f'Error opening file: {e}'},
        )
    except httpx.TimeoutException:
        logger.error(f'Timeout reading file for conversation {conversation_id}')
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={'error': 'Request to runtime timed out'},
        )
    except httpx.ConnectError:
        logger.error(
            f'Connection error reading file for conversation {conversation_id}'
        )
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={'error': 'Unable to connect to runtime'},
        )
    except Exception as e:
        logger.error(f'Error opening file {file}: {e}')
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': f'Error opening file: {e}'},
        )

    if content is not None:
        return JSONResponse(content={'code': content})
    elif error and error.startswith('BINARY_FILE:'):
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content={'error': f'Unable to open binary file: {file}'},
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': f'Error opening file: {error}'},
        )


@app.get(
    '/zip-directory',
    response_model=None,
    responses={
        200: {'description': 'Zipped workspace returned as FileResponse'},
        500: {'description': 'Error zipping workspace', 'model': dict},
    },
    deprecated=True,
)
def zip_current_workspace(
    conversation: ServerConversation = Depends(get_conversation),
) -> FileResponse | JSONResponse:
    """Download the current workspace as a zip file.

    For V1 conversations, file operations are handled through the agent server.
    Use the sandbox's exposed agent server URL to access file operations.
    """
    try:
        logger.debug('Zipping workspace')
        runtime: Runtime = conversation.runtime
        path = runtime.config.workspace_mount_path_in_sandbox
        try:
            zip_file_path = runtime.copy_from(path)
        except AgentRuntimeUnavailableError as e:
            logger.error(f'Error zipping workspace: {e}')
            return JSONResponse(
                status_code=500,
                content={'error': f'Error zipping workspace: {e}'},
            )
        return FileResponse(
            path=zip_file_path,
            filename='workspace.zip',
            media_type='application/zip',
            background=BackgroundTask(lambda: os.unlink(zip_file_path)),
        )
    except Exception as e:
        logger.error(f'Error zipping workspace: {e}')
        raise HTTPException(
            status_code=500,
            detail='Failed to zip workspace',
        )


@app.get(
    '/git/changes',
    response_model=list[dict[str, str]],
    responses={
        404: {'description': 'Not a git repository', 'model': dict},
        500: {'description': 'Error getting changes', 'model': dict},
    },
    deprecated=True,
)
async def git_changes(
    conversation: ServerConversation = Depends(get_conversation),
    conversation_store: ConversationStore = Depends(get_conversation_store),
    user_id: str = Depends(get_user_id),
) -> list[dict[str, str]] | JSONResponse:
    """Get git changes in the workspace.

    For V1 conversations, git operations are handled through the agent server.
    Use the sandbox's exposed agent server URL to access git operations.
    """
    runtime: Runtime = conversation.runtime

    cwd = runtime.config.workspace_mount_path_in_sandbox
    logger.info(f'Getting git changes in {cwd}')

    try:
        changes = await call_sync_from_async(runtime.get_git_changes, cwd)
        if changes is None:
            return JSONResponse(
                status_code=404,
                content={'error': 'Not a git repository'},
            )
        return changes
    except AgentRuntimeUnavailableError as e:
        logger.error(f'Runtime unavailable: {e}')
        return JSONResponse(
            status_code=500,
            content={'error': f'Error getting changes: {e}'},
        )
    except Exception as e:
        logger.error(f'Error getting changes: {e}')
        return JSONResponse(
            status_code=500,
            content={'error': str(e)},
        )


@app.get(
    '/git/diff',
    response_model=dict[str, Any],
    responses={500: {'description': 'Error getting diff', 'model': dict}},
    deprecated=True,
)
async def git_diff(
    path: str,
    conversation_store: Any = Depends(get_conversation_store),
    conversation: ServerConversation = Depends(get_conversation),
) -> dict[str, Any] | JSONResponse:
    """Get git diff for a specific file.

    For V1 conversations, git operations are handled through the agent server.
    Use the sandbox's exposed agent server URL to access git operations.
    """
    runtime: Runtime = conversation.runtime

    cwd = runtime.config.workspace_mount_path_in_sandbox

    try:
        diff = await call_sync_from_async(runtime.get_git_diff, path, cwd)
        return diff
    except AgentRuntimeUnavailableError as e:
        logger.error(f'Error getting diff: {e}')
        return JSONResponse(
            status_code=500,
            content={'error': f'Error getting diff: {e}'},
        )


@app.post('/upload-files', response_model=POSTUploadFilesModel, deprecated=True)
async def upload_files(
    files: list[UploadFile],
    metadata: ConversationMetadata = Depends(get_conversation_metadata),
):
    """Upload files to the workspace.

    For V1 conversations, file operations are handled through the agent server.
    Use the sandbox's exposed agent server URL to access file operations.
    """
    conversation_id = metadata.conversation_id

    # Read all file contents
    file_data: list[tuple[str, bytes]] = []
    for file in files:
        content = await file.read()
        file_data.append((str(file.filename), content))

    try:
        uploaded_files, skipped_files = await conversation_manager.upload_files(
            conversation_id, file_data
        )
    except ValueError as e:
        logger.error(f'Error uploading files: {e}')
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={'error': str(e)},
        )
    except httpx.TimeoutException:
        logger.error(f'Timeout uploading files for conversation {conversation_id}')
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={'error': 'Request to runtime timed out'},
        )
    except httpx.ConnectError:
        logger.error(
            f'Connection error uploading files for conversation {conversation_id}'
        )
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={'error': 'Unable to connect to runtime'},
        )
    except httpx.HTTPStatusError as e:
        logger.error(f'HTTP error uploading files: {e.response.status_code}')
        return JSONResponse(
            status_code=e.response.status_code,
            content={'error': f'Runtime returned error: {e.response.status_code}'},
        )
    except Exception as e:
        logger.error(f'Error uploading files: {e}')
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': f'Error uploading files: {e}'},
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            'uploaded_files': uploaded_files,
            'skipped_files': skipped_files,
        },
    )
