"""Secret sources and types for handling sensitive data."""

from abc import ABC, abstractmethod

import httpx
from pydantic import Field, SecretStr, field_serializer, field_validator

from openhands.sdk.utils.models import DiscriminatedUnionMixin
from openhands.sdk.utils.pydantic_secrets import serialize_secret, validate_secret


class SecretSource(DiscriminatedUnionMixin, ABC):
    """Source for a named secret which may be obtained dynamically"""

    description: str | None = Field(
        default=None,
        description="Optional description for this secret",
    )

    @abstractmethod
    def get_value(self) -> str | None:
        """Get the value of a secret in plain text"""


class StaticSecret(SecretSource):
    """A secret stored locally"""

    value: SecretStr

    def get_value(self):
        return self.value.get_secret_value()

    @field_validator("value")
    @classmethod
    def _validate_secrets(cls, v: SecretStr | None, info):
        return validate_secret(v, info)

    @field_serializer("value", when_used="always")
    def _serialize_secrets(self, v: SecretStr | None, info):
        return serialize_secret(v, info)


class LookupSecret(SecretSource):
    """A secret looked up from some external url"""

    url: str
    headers: dict[str, str] = Field(default_factory=dict)

    def get_value(self):
        response = httpx.get(self.url, headers=self.headers, timeout=30.0)
        response.raise_for_status()
        return response.text

    @field_validator("headers")
    @classmethod
    def _validate_secrets(cls, headers: dict[str, str], info):
        result = {}
        for key, value in headers.items():
            if _is_secret_header(key):
                secret_value = validate_secret(SecretStr(value), info)
                assert secret_value is not None
                result[key] = secret_value.get_secret_value()
            else:
                result[key] = value
        return result

    @field_serializer("headers", when_used="always")
    def _serialize_secrets(self, headers: dict[str, str], info):
        result = {}
        for key, value in headers.items():
            if _is_secret_header(key):
                secret_value = serialize_secret(SecretStr(value), info)
                assert secret_value is not None
                result[key] = secret_value
            else:
                result[key] = value
        return result


_SECRET_HEADERS = ["AUTHORIZATION", "KEY", "SECRET"]


def _is_secret_header(key: str):
    key = key.upper()
    for secret in _SECRET_HEADERS:
        if secret in key:
            return True
    return False


# Type alias for secret values - can be a plain string or a SecretSource
SecretValue = str | SecretSource
