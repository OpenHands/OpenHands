"""Credential mapping models for resource-based authentication."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CredentialMapping(BaseModel):
    """Maps a resource pattern to authentication credentials.

    This allows OpenHands to automatically authenticate with private resources
    at runtime by matching resource URLs/domains to stored credentials.

    Attributes:
        resource_pattern: URL pattern or domain to match (e.g., "npm.example.com",
            "https://api.example.com/*", "*.private-registry.io")
        credential_name: Name of the custom secret that contains the credential value
        auth_method: How to use the credential for authentication
        auth_header: Custom header name (required for "header" method, optional otherwise)
        resource_type: Optional type identifier (e.g., "npm", "docker", "api", "database")
        description: Optional description of this credential mapping
    """

    resource_pattern: str = Field(
        ...,
        description='URL pattern or domain to match (e.g., "npm.example.com", "https://api.example.com/*")',
    )
    credential_name: str = Field(
        ...,
        description='Name of the custom secret that contains the credential value',
    )
    auth_method: Literal['api_key', 'bearer_token', 'basic_auth', 'header'] = Field(
        ...,
        description='How to use the credential for authentication',
    )
    auth_header: str | None = Field(
        None,
        description='Custom header name (required for "header" method)',
    )
    resource_type: str | None = Field(
        None,
        description='Optional type identifier (e.g., "npm", "docker", "api")',
    )
    description: str | None = Field(None, description='Optional description')

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
    )

    @model_validator(mode='after')
    def validate_auth_method(self) -> 'CredentialMapping':
        """Validate that auth_header is provided when auth_method is 'header'."""
        if self.auth_method == 'header' and not self.auth_header:
            raise ValueError('auth_header is required when auth_method is "header"')
        return self
