# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class SecurityConfig(BaseModel):
    """Configuration for security related functionalities.

    Attributes:
        confirmation_mode: Whether to enable confirmation mode.
        security_analyzer: The security analyzer to use.
    """

    confirmation_mode: bool = Field(default=False)
    security_analyzer: str | None = Field(default=None)

    model_config = ConfigDict(extra='forbid')

    @classmethod
    def from_toml_section(cls, data: dict) -> dict[str, 'SecurityConfig']:
        """Create a mapping of SecurityConfig instances from a toml dictionary representing the [security] section.

        The configuration is built from all keys in data.

        Returns:
            dict[str, SecurityConfig]: A mapping where the key "security" corresponds to the [security] configuration
        """
        # Initialize the result mapping
        security_mapping: dict[str, SecurityConfig] = {}

        # Try to create the configuration instance
        try:
            security_mapping['security'] = cls.model_validate(data)
        except ValidationError as e:
            raise ValueError(f'Invalid security configuration: {e}')

        return security_mapping
