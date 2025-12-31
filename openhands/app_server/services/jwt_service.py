import hashlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncGenerator

import jwt
from fastapi import Request
from jose import jwe
from jose.constants import ALGORITHMS
from pydantic import BaseModel, PrivateAttr

from openhands.agent_server.utils import utc_now
from openhands.app_server.services.injector import Injector, InjectorState
from openhands.app_server.utils.encryption_key import (
    EncryptionKey,
    get_default_encryption_keys,
)


class JwtService:
    """Service for signing/verifying JWS tokens and encrypting/decrypting JWE tokens."""

    def __init__(self, keys: list[EncryptionKey]):
        """Initialize the JWT service with a list of keys.

        Args:
            keys: List of EncryptionKey objects. If None, will try to load from config.

        Raises:
            ValueError: If no keys are provided and config is not available
        """
        active_keys = [key for key in keys if key.active]
        if not active_keys:
            raise ValueError('At least one active key is required')

        # Store keys by ID for quick lookup
        self._keys = {key.id: key for key in keys}

        # Find the newest key as default
        newest_key = max(active_keys, key=lambda k: k.created_at)
        self._default_key_id = newest_key.id

    @property
    def default_key_id(self) -> str:
        """Get the default key ID."""
        return self._default_key_id

    def create_jws_token(
        self,
        payload: dict[str, Any],
        key_id: str | None = None,
        expires_in: timedelta | None = None,
    ) -> str:
        """Create a JWS (JSON Web Signature) token.

        Args:
            payload: The JWT payload
            key_id: The key ID to use for signing. If None, uses the newest key.
            expires_in: Token expiration time. If None, defaults to 1 hour.

        Returns:
            The signed JWS token

        Raises:
            ValueError: If key_id is invalid
        """
        if key_id is None:
            key_id = self._default_key_id

        if key_id not in self._keys:
            raise ValueError(f"Key ID '{key_id}' not found")

        # Add standard JWT claims
        now = utc_now()
        if expires_in is None:
            expires_in = timedelta(hours=1)

        jwt_payload = {
            **payload,
            'iat': int(now.timestamp()),
            'exp': int((now + expires_in).timestamp()),
        }

        # Use the raw key for JWT signing with key_id in header
        secret_key = self._keys[key_id].key.get_secret_value()

        return jwt.encode(
            jwt_payload, secret_key, algorithm='HS256', headers={'kid': key_id}
        )

    def verify_jws_token(self, token: str, key_id: str | None = None) -> dict[str, Any]:
        """Verify and decode a JWS token.

        Args:
            token: The JWS token to verify
            key_id: The key ID to use for verification. If None, extracts from
                    token's kid header.

        Returns:
            The decoded JWT payload

        Raises:
            ValueError: If token is invalid or key_id is not found
            jwt.InvalidTokenError: If token verification fails
        """
        if key_id is None:
            # Try to extract key_id from the token's kid header
            try:
                unverified_header = jwt.get_unverified_header(token)
                key_id = unverified_header.get('kid')
                if not key_id:
                    raise ValueError("Token does not contain 'kid' header with key ID")
            except jwt.DecodeError:
                raise ValueError('Invalid JWT token format')

        if key_id not in self._keys:
            raise ValueError(f"Key ID '{key_id}' not found")

        # Use the raw key for JWT verification
        secret_key = self._keys[key_id].key.get_secret_value()

        try:
            payload = jwt.decode(token, secret_key, algorithms=['HS256'])
            return payload
        except jwt.InvalidTokenError as e:
            raise jwt.InvalidTokenError(f'Token verification failed: {str(e)}')

    def create_jwe_token(
        self,
        payload: dict[str, Any],
        key_id: str | None = None,
        expires_in: timedelta | None = None,
    ) -> str:
        """Create a JWE (JSON Web Encryption) token.

        Args:
            payload: The JWT payload to encrypt
            key_id: The key ID to use for encryption. If None, uses the newest key.
            expires_in: Token expiration time. If None, defaults to 1 hour.

        Returns:
            The encrypted JWE token

        Raises:
            ValueError: If key_id is invalid
        """
        if key_id is None:
            key_id = self._default_key_id

        if key_id not in self._keys:
            raise ValueError(f"Key ID '{key_id}' not found")

        # Add standard JWT claims
        now = utc_now()
        if expires_in is None:
            expires_in = timedelta(hours=1)

        jwt_payload = {
            **payload,
            'iat': int(now.timestamp()),
            'exp': int((now + expires_in).timestamp()),
        }

        # Get the raw key for JWE encryption and derive a 256-bit key
        secret_key = self._keys[key_id].key.get_secret_value()
        key_bytes = secret_key.encode() if isinstance(secret_key, str) else secret_key
        # Derive a 256-bit key using SHA256
        key_256 = hashlib.sha256(key_bytes).digest()

        # Encrypt the payload (convert to JSON string first)
        payload_json = json.dumps(jwt_payload)
        encrypted_token = jwe.encrypt(
            payload_json,
            key_256,
            algorithm=ALGORITHMS.DIR,
            encryption=ALGORITHMS.A256GCM,
            kid=key_id,
        )
        # Ensure we return a string
        return (
            encrypted_token.decode('utf-8')
            if isinstance(encrypted_token, bytes)
            else encrypted_token
        )

    def decrypt_jwe_token(
        self, token: str, key_id: str | None = None
    ) -> dict[str, Any]:
        """Decrypt and decode a JWE token.

        Args:
            token: The JWE token to decrypt
            key_id: The key ID to use for decryption. If None, extracts
                    from token header.

        Returns:
            The decrypted JWT payload

        Raises:
            ValueError: If token is invalid or key_id is not found
            Exception: If token decryption fails
        """
        if key_id is None:
            # Try to extract key_id from the token's header
            try:
                header = jwe.get_unverified_header(token)
                key_id = header.get('kid')
                if not key_id:
                    raise ValueError("Token does not contain 'kid' header with key ID")
            except Exception:
                raise ValueError('Invalid JWE token format')

        if key_id not in self._keys:
            raise ValueError(f"Key ID '{key_id}' not found")

        # Get the raw key for JWE decryption and derive a 256-bit key
        secret_key = self._keys[key_id].key.get_secret_value()
        key_bytes = secret_key.encode() if isinstance(secret_key, str) else secret_key
        # Derive a 256-bit key using SHA256
        key_256 = hashlib.sha256(key_bytes).digest()

        try:
            payload_json = jwe.decrypt(token, key_256)
            assert payload_json is not None
            # Parse the JSON string back to dictionary
            payload = json.loads(payload_json)
            return payload
        except Exception as e:
            raise Exception(f'Token decryption failed: {str(e)}')


class JwtServiceInjector(BaseModel, Injector[JwtService]):
    persistence_dir: Path
    _jwt_service: JwtService | None = PrivateAttr(default=None)

    def get_jwt_service(self) -> JwtService:
        jwt_service = self._jwt_service
        if jwt_service is None:
            keys = get_default_encryption_keys(self.persistence_dir)
            jwt_service = JwtService(keys=keys)
            self._jwt_service = jwt_service
        return jwt_service

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[JwtService, None]:
        yield self.get_jwt_service()
