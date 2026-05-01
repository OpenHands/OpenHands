import datetime
import hashlib
import os
from pathlib import Path
from typing import Any

import base62
from pydantic import BaseModel, Field, SecretStr, TypeAdapter, field_serializer

from openhands.agent_server.utils import utc_now


class EncryptionKey(BaseModel):
    """Configuration for an encryption key."""

    id: str = Field(default_factory=lambda: base62.encodebytes(os.urandom(32)))
    key: SecretStr
    active: bool = True
    notes: str | None = None
    created_at: datetime.datetime = Field(default_factory=utc_now)

    @field_serializer('key')
    def serialize_key(self, key: SecretStr, info: Any):
        """Conditionally serialize the key based on context."""
        if info.context and info.context.get('expose_secrets'):
            return key.get_secret_value()
        return str(key)  # Returns '**********' by default


def get_default_encryption_keys(workspace_dir: Path) -> list[EncryptionKey]:
    """Generate default encryption keys obtain these from previously saved values
    and environment variables."""
    encryption_keys = []

    jwt_secret = os.getenv('JWT_SECRET')
    if jwt_secret:
        # Derive a deterministic key ID from the secret itself.
        # This ensures all pods using the same JWT_SECRET get the same key ID,
        # which is critical for multi-pod deployments where tokens may be
        # created by one pod and verified by another.
        key_id = base62.encodebytes(hashlib.sha256(jwt_secret.encode()).digest())
        return [
            EncryptionKey(
                id=key_id,
                key=SecretStr(jwt_secret),
                active=True,
                notes='jwt secret master key',
            )
        ]

    key_file = workspace_dir / '.keys'
    type_adapter = TypeAdapter(list[EncryptionKey])
    if key_file.exists():
        encryption_keys = type_adapter.validate_json(key_file.read_text())

    # Fallback to JWTSecret...
    jwt_secret_file = workspace_dir / '.jwt_secret'
    if jwt_secret_file.exists():
        jwt_secret = jwt_secret_file.read_text().strip()
        encryption_keys.append(
            EncryptionKey(
                id=base62.encodebytes(hashlib.sha256(jwt_secret.encode()).digest()),
                key=SecretStr(jwt_secret),
                active=True,
                notes='jwt secret master key',
                created_at=datetime.datetime.fromtimestamp(
                    jwt_secret_file.stat().st_mtime, tz=datetime.UTC
                ),
            )
        )

    if encryption_keys:
        return encryption_keys

    encryption_keys = [
        EncryptionKey(
            key=SecretStr(base62.encodebytes(os.urandom(32))),
            active=True,
            notes='generated master key',
        )
    ]
    json_data = type_adapter.dump_json(
        encryption_keys, context={'expose_secrets': True}
    )
    key_file.write_bytes(json_data)
    return encryption_keys
