"""SSH Keys router for OpenHands App Server.

This module provides the V1 API routes for SSH public keys under /api/v1/ssh-keys.
SSH public keys are used for passwordless SSH access to sandbox containers.
"""

import re
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from openhands.app_server.utils.dependencies import get_dependencies
from openhands.app_server.utils.models import EditResponse
from openhands.server.user_auth import get_user_settings, get_user_settings_store
from openhands.storage.data_models.settings import SSHPublicKey, Settings
from openhands.storage.settings.settings_store import SettingsStore

# Create router with /api/v1/ssh-keys prefix
router = APIRouter(
    prefix='/ssh-keys',
    tags=['SSH Keys'],
    dependencies=get_dependencies(),
)


# SSH public key validation regex
# Matches common SSH key formats: ssh-rsa, ssh-ed25519, ecdsa-sha2-*, etc.
SSH_KEY_PATTERN = re.compile(
    r'^(ssh-rsa|ssh-ed25519|ssh-dss|ecdsa-sha2-nistp256|ecdsa-sha2-nistp384|ecdsa-sha2-nistp521|sk-ssh-ed25519@openssh\.com|sk-ecdsa-sha2-nistp256@openssh\.com)\s+[A-Za-z0-9+/=]+(\s+.*)?$'
)


class SSHKeyRequest(BaseModel):
    """Request model for adding/updating an SSH public key."""

    key: str = Field(
        description='The SSH public key (e.g., ssh-ed25519 AAAA... user@host)'
    )
    label: str | None = Field(
        default=None, description='Optional label for the key (e.g., "Work Laptop")'
    )

    @field_validator('key')
    @classmethod
    def validate_ssh_key(cls, v: str) -> str:
        v = v.strip()
        if not SSH_KEY_PATTERN.match(v):
            raise ValueError(
                'Invalid SSH public key format. '
                'Expected format: <key-type> <base64-key> [comment]'
            )
        return v


class SSHKeyResponse(BaseModel):
    """Response model for an SSH public key."""

    id: str = Field(description='Unique identifier for the key')
    key: str = Field(description='The SSH public key')
    label: str | None = Field(description='Optional label for the key')


class SSHKeysListResponse(BaseModel):
    """Response model for listing SSH public keys."""

    ssh_keys: list[SSHKeyResponse]


def _generate_key_id(key: str) -> str:
    """Generate a unique ID for an SSH key based on its content."""
    # Use a hash of the key as the ID to make it deterministic
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def _settings_to_key_responses(settings: Settings | None) -> list[SSHKeyResponse]:
    """Convert settings SSH keys to response format with IDs."""
    if not settings or not settings.ssh_public_keys:
        return []

    return [
        SSHKeyResponse(
            id=_generate_key_id(ssh_key.key),
            key=ssh_key.key,
            label=ssh_key.label,
        )
        for ssh_key in settings.ssh_public_keys
    ]


@router.get('', response_model=SSHKeysListResponse)
async def list_ssh_keys(
    settings: Settings | None = Depends(get_user_settings),
) -> SSHKeysListResponse:
    """List all SSH public keys.

    Retrieves all SSH public keys for the authenticated user.

    Returns:
        SSHKeysListResponse: List of SSH public keys with their IDs and labels
    """
    return SSHKeysListResponse(ssh_keys=_settings_to_key_responses(settings))


@router.post('', status_code=status.HTTP_201_CREATED, response_model=SSHKeyResponse)
async def add_ssh_key(
    request: SSHKeyRequest,
    settings_store: SettingsStore | None = Depends(get_user_settings_store),
) -> SSHKeyResponse:
    """Add a new SSH public key.

    Adds a new SSH public key for passwordless SSH access to sandboxes.

    Returns:
        201: SSH key added successfully
        400: Invalid key format or key already exists
        500: Error adding SSH key
    """
    if settings_store is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='User authentication required',
        )
    existing_settings = await settings_store.load()
    ssh_keys = list(existing_settings.ssh_public_keys) if existing_settings else []

    # Check if key already exists
    for existing_key in ssh_keys:
        if existing_key.key == request.key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='This SSH key already exists',
            )

    # Add new key
    new_key = SSHPublicKey(key=request.key, label=request.label)
    ssh_keys.append(new_key)

    # Update settings
    if existing_settings:
        updated_settings = existing_settings.model_copy(
            update={'ssh_public_keys': ssh_keys}
        )
    else:
        updated_settings = Settings(ssh_public_keys=ssh_keys)

    await settings_store.store(updated_settings)

    return SSHKeyResponse(
        id=_generate_key_id(request.key),
        key=request.key,
        label=request.label,
    )


@router.put('/{key_id}', response_model=SSHKeyResponse)
async def update_ssh_key(
    key_id: str,
    request: SSHKeyRequest,
    settings_store: SettingsStore | None = Depends(get_user_settings_store),
) -> Any:
    """Update an existing SSH public key.

    Updates the key content and/or label of an existing SSH key.

    Returns:
        200: SSH key updated successfully
        400: Invalid key format or key already exists
        404: SSH key not found
        500: Error updating SSH key
    """
    if settings_store is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='User authentication required',
        )
    existing_settings = await settings_store.load()
    if not existing_settings or not existing_settings.ssh_public_keys:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'SSH key with ID {key_id} not found',
        )

    ssh_keys = list(existing_settings.ssh_public_keys)

    # Find the key to update
    key_index = None
    for i, ssh_key in enumerate(ssh_keys):
        if _generate_key_id(ssh_key.key) == key_id:
            key_index = i
            break

    if key_index is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'SSH key with ID {key_id} not found',
        )

    # Check if the new key already exists (if changing the key content)
    if request.key != ssh_keys[key_index].key:
        for i, existing_key in enumerate(ssh_keys):
            if i != key_index and existing_key.key == request.key:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail='This SSH key already exists',
                )

    # Update the key
    ssh_keys[key_index] = SSHPublicKey(key=request.key, label=request.label)

    # Update settings
    updated_settings = existing_settings.model_copy(
        update={'ssh_public_keys': ssh_keys}
    )
    await settings_store.store(updated_settings)

    return SSHKeyResponse(
        id=_generate_key_id(request.key),
        key=request.key,
        label=request.label,
    )


@router.delete('/{key_id}')
async def delete_ssh_key(
    key_id: str,
    settings_store: SettingsStore | None = Depends(get_user_settings_store),
) -> EditResponse:
    """Delete an SSH public key.

    Removes an SSH public key for the authenticated user.

    Returns:
        200: SSH key deleted successfully
        404: SSH key not found
        500: Error deleting SSH key
    """
    if settings_store is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='User authentication required',
        )
    existing_settings = await settings_store.load()
    if not existing_settings or not existing_settings.ssh_public_keys:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'SSH key with ID {key_id} not found',
        )

    ssh_keys = list(existing_settings.ssh_public_keys)

    # Find and remove the key
    key_found = False
    for i, ssh_key in enumerate(ssh_keys):
        if _generate_key_id(ssh_key.key) == key_id:
            ssh_keys.pop(i)
            key_found = True
            break

    if not key_found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'SSH key with ID {key_id} not found',
        )

    # Update settings
    updated_settings = existing_settings.model_copy(
        update={'ssh_public_keys': ssh_keys}
    )
    await settings_store.store(updated_settings)

    return EditResponse(message='SSH key deleted successfully')
