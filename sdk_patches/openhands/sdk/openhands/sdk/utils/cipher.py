"""
Cipher utility for preventing accidental secret disclosure in serialized data

SECURITY WARNINGS:
- The secret key is a string for ease of use but should contain at least 256
  bits of entropy
"""

import hashlib
from base64 import b64encode

from cryptography.fernet import Fernet
from pydantic import SecretStr


class Cipher:
    """
    Simple encryption utility for preventing accidental secret disclosure.
    """

    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self._fernet: Fernet | None = None

    def encrypt(self, secret: SecretStr | None) -> str | None:
        if secret is None:
            return None
        secret_value = secret.get_secret_value().encode()
        fernet = self._get_fernet()
        result = fernet.encrypt(secret_value).decode()
        return result

    def decrypt(self, secret: str | None) -> SecretStr | None:
        """
        Decrypt a secret value, returning None if decryption fails.

        This handles cases where existing conversations were serialized with different
        encryption keys or contain invalid encrypted data. A warning is logged when
        decryption fails and a None is returned. This mimics the case where
        no cipher was defined so secrets where redacted.
        """
        if secret is None:
            return None
        try:
            fernet = self._get_fernet()
            decrypted = fernet.decrypt(secret.encode()).decode()
            return SecretStr(decrypted)
        except Exception as e:
            # Import here to avoid circular imports
            from openhands.sdk.logger import get_logger

            logger = get_logger(__name__)
            logger.warning(
                f"Failed to decrypt secret value (setting to None): {e}. "
                "This may occur when loading conversations encrypted with a different "
                "key or when upgrading from older versions."
            )
            return None

    def _get_fernet(self):
        fernet = self._fernet
        if fernet is None:
            secret_key = self.secret_key.encode()
            # Hash the key to make sure we have a 256 bit value
            fernet_key = b64encode(hashlib.sha256(secret_key).digest())
            fernet = Fernet(fernet_key)
            object.__setattr__(self, "_fernet", fernet)
        return fernet
