import binascii
import hashlib
import json
import logging
from base64 import b64decode, b64encode
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncGenerator

import jwt
from cryptography.fernet import Fernet, InvalidToken
from fastapi import Request
from jwcrypto import jwe as jwcrypto_jwe
from jwcrypto import jwk
from pydantic import BaseModel, PrivateAttr

from openhands.agent_server.utils import utc_now
from openhands.app_server.services.injector import Injector, InjectorState
from openhands.app_server.utils.encryption_key import (
    EncryptionKey,
    get_default_encryption_keys,
)

logger = logging.getLogger(__name__)


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

    @property
    def key_ids(self) -> list[str]:
        return list(self._keys)

    def get_key(self, key_id: str) -> EncryptionKey:
        return self._keys[key_id]

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
                    # Legacy tokens created before key rotation support
                    # don't carry a kid header — fall back to the default key.
                    key_id = self._default_key_id
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
        jwt_payload = {
            **payload,
            'iat': int(now.timestamp()),
        }

        # Only add exp if expires_in is provided
        if expires_in is not None:
            jwt_payload['exp'] = int((now + expires_in).timestamp())

        # Get the raw key for JWE encryption and derive a 256-bit key
        secret_key = self._keys[key_id].key.get_secret_value()
        key_bytes = secret_key.encode() if isinstance(secret_key, str) else secret_key
        # Derive a 256-bit key using SHA256
        key_256 = hashlib.sha256(key_bytes).digest()

        # Create JWK from symmetric key for jwcrypto
        symmetric_key = jwk.JWK(kty='oct', k=jwk.base64url_encode(key_256))

        # Create JWE token with jwcrypto
        protected_header = {
            'alg': 'dir',
            'enc': 'A256GCM',
            'kid': key_id,
        }
        jwe_token = jwcrypto_jwe.JWE(
            json.dumps(jwt_payload).encode('utf-8'),
            recipient=symmetric_key,
            protected=protected_header,
        )
        return jwe_token.serialize(compact=True)

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
        # Deserialize once and reuse for both header extraction and decryption
        try:
            jwe_obj = jwcrypto_jwe.JWE()
            jwe_obj.deserialize(token)
        except Exception:
            raise ValueError('Invalid JWE token format')

        # Extract and validate the protected header
        try:
            protected_header = json.loads(jwe_obj.objects['protected'])
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f'Invalid JWE token format: {type(e).__name__}')

        # Verify algorithms to prevent cryptographic agility attacks
        if (
            protected_header.get('alg') != 'dir'
            or protected_header.get('enc') != 'A256GCM'
        ):
            raise ValueError('Unsupported or unexpected JWE algorithm')

        if key_id is None:
            # Extract key_id from the token's header
            key_id = protected_header.get('kid')
            if not key_id:
                raise ValueError("Token does not contain 'kid' header with key ID")

        if key_id not in self._keys:
            raise ValueError(f"Key ID '{key_id}' not found")

        # Get the raw key for JWE decryption and derive a 256-bit key
        secret_key = self._keys[key_id].key.get_secret_value()
        key_bytes = secret_key.encode() if isinstance(secret_key, str) else secret_key
        # Derive a 256-bit key using SHA256
        key_256 = hashlib.sha256(key_bytes).digest()

        try:
            # Create JWK from symmetric key for jwcrypto
            symmetric_key = jwk.JWK(kty='oct', k=jwk.base64url_encode(key_256))

            # Decrypt the JWE token (reusing already deserialized jwe_obj)
            jwe_obj.decrypt(symmetric_key)

            # Parse the JSON string back to dictionary
            payload = json.loads(jwe_obj.payload.decode('utf-8'))
            return payload
        except Exception as e:
            raise Exception(f'Token decryption failed: {str(e)}')

    # ------------------------------------------------------------------
    # Symmetric encrypt / decrypt helpers (JWE with legacy Fernet fallback)
    # ------------------------------------------------------------------

    def encrypt_value(self, plaintext: str) -> str:
        """Encrypt a plaintext string using JWE.

        New data is always encrypted with JWE. Use :meth:`decrypt_value`
        to decrypt, which also handles legacy Fernet-encrypted data.
        """
        return self.create_jwe_token({'v': plaintext})

    def decrypt_value(self, ciphertext: str) -> str:
        """Decrypt a ciphertext string, trying JWE first then legacy Fernet.

        During the migration from Fernet to JWE, persisted data may be
        encrypted with either scheme.  This method transparently handles
        both: it attempts JWE decryption first, and falls back to Fernet
        (trying every known key) if that fails.
        """
        # Try JWE first (modern path)
        try:
            payload = self.decrypt_jwe_token(ciphertext)
            return payload['v']
        except Exception:
            pass

        # Fall back to legacy Fernet decryption
        return self._decrypt_fernet_value(ciphertext)

    def _decrypt_fernet_value(self, ciphertext: str) -> str:
        """Attempt Fernet decryption using every known key.

        The legacy Fernet key is derived as
        ``b64encode(sha256(secret).digest())``, matching the convention
        previously used by enterprise code.

        Some values were base64 encoded after encryption, and some were not,
        so we accommodate both cases.

        Raises ``ValueError`` if no key can decrypt the value.
        """
        last_error: Exception | None = None
        for key in self._keys.values():
            try:
                secret = key.key.get_secret_value()
                fernet_key = b64encode(hashlib.sha256(secret.encode()).digest())
                f = Fernet(fernet_key)
                # There are multiple legacy formats - some cases have base64 encoded
                # after encryption and some have not. We try both
                try:
                    return f.decrypt(b64decode(ciphertext.encode())).decode()
                except Exception:
                    return f.decrypt(ciphertext.encode()).decode()
            except (InvalidToken, binascii.Error, Exception) as exc:
                last_error = exc
                continue

        raise ValueError('Failed to decrypt value with any known key') from last_error


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
