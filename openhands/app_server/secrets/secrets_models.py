"""Models for the secrets API."""

from pydantic import BaseModel, SecretStr


class CustomSecretWithoutValue(BaseModel):
    """Custom secret model without value (for listing secrets)."""

    name: str
    description: str | None = None


class CustomSecretCreate(CustomSecretWithoutValue):
    """Custom secret model with value (for creating secrets)."""

    value: SecretStr


class CustomSecretPage(BaseModel):
    """Paginated response for custom secrets search."""

    items: list[CustomSecretWithoutValue]
    next_page_id: str | None = None
