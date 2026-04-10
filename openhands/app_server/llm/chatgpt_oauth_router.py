"""HTTP routes for ChatGPT subscription device OAuth on the LLM settings flow."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, SecretStr

from openhands.app_server.llm.chatgpt_oauth import (
    CHATGPT_DEVICE_VERIFY_URL,
    exchange_authorization_code,
    poll_once,
    request_device_code,
)
from openhands.app_server.utils.dependencies import get_dependencies
from openhands.core.logger import openhands_logger as logger
from openhands.server.user_auth import get_user_id, get_user_settings_store
from openhands.storage.data_models.settings import Settings
from openhands.storage.settings.settings_store import SettingsStore
from openhands.utils.chatgpt_oauth_tokens import encode_chatgpt_token_bundle

router = APIRouter(
    prefix='/llm/chatgpt',
    tags=['ChatGPT LLM'],
    dependencies=get_dependencies(),
)

DEFAULT_CHATGPT_MODEL = 'chatgpt/gpt-5.2-codex'

_sessions: dict[str, dict[str, Any]] = {}
_session_lock = asyncio.Lock()


class DeviceSessionResponse(BaseModel):
    session_id: str
    user_code: str
    verification_uri: str


class PollResponse(BaseModel):
    status: str


@router.post('/device-session', response_model=DeviceSessionResponse)
async def create_device_session(
    user_id: str | None = Depends(get_user_id),
) -> DeviceSessionResponse:
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Authentication required',
        )
    try:
        device = await asyncio.to_thread(request_device_code)
    except Exception as e:
        logger.warning('ChatGPT device code request failed: %s', e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail='Could not start ChatGPT sign-in',
        ) from e
    session_id = str(uuid.uuid4())
    async with _session_lock:
        _sessions[session_id] = {
            'user_id': user_id,
            'device_auth_id': device['device_auth_id'],
            'user_code': device['user_code'],
            'interval': int(device.get('interval', '5')),
        }
    return DeviceSessionResponse(
        session_id=session_id,
        user_code=device['user_code'],
        verification_uri=CHATGPT_DEVICE_VERIFY_URL,
    )


@router.get('/poll/{session_id}', response_model=PollResponse)
async def poll_device_session(
    session_id: str,
    user_id: str | None = Depends(get_user_id),
    settings_store: SettingsStore | None = Depends(get_user_settings_store),
) -> PollResponse:
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Authentication required',
        )
    if settings_store is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Settings store unavailable',
        )
    async with _session_lock:
        sess = _sessions.get(session_id)
    if not sess or sess.get('user_id') != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Unknown session',
        )
    try:
        raw = await asyncio.to_thread(
            poll_once,
            sess['device_auth_id'],
            sess['user_code'],
        )
    except Exception as e:
        logger.warning('ChatGPT device poll failed: %s', e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail='ChatGPT sign-in poll failed',
        ) from e
    if raw is None:
        return PollResponse(status='pending')
    try:
        tokens = await asyncio.to_thread(exchange_authorization_code, raw)
    except Exception as e:
        logger.warning('ChatGPT token exchange failed: %s', e)
        async with _session_lock:
            _sessions.pop(session_id, None)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail='Could not complete ChatGPT sign-in',
        ) from e

    existing = await settings_store.load()
    if not existing:
        async with _session_lock:
            _sessions.pop(session_id, None)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Settings not found',
        )
    bundle = encode_chatgpt_token_bundle(
        tokens['access_token'],
        tokens['refresh_token'],
    )
    merged_data = existing.model_dump(context={'expose_secrets': True})
    merged_data['llm_model'] = DEFAULT_CHATGPT_MODEL
    merged_data['llm_api_key'] = SecretStr(bundle)
    merged_data['llm_base_url'] = None
    await settings_store.store(Settings(**merged_data))
    async with _session_lock:
        _sessions.pop(session_id, None)
    return PollResponse(status='complete')
