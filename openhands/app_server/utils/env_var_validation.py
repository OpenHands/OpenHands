"""Utilities for validating environment variable names."""

import re

# Must start with a letter or underscore, contain only alphanumeric characters and underscores.
ENV_VAR_NAME_PATTERN = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')


def is_valid_env_var_name(name: str) -> bool:
    """Check if a name is valid for use as an environment variable."""
    return bool(ENV_VAR_NAME_PATTERN.fullmatch(name))


def validate_env_var_name(name: str, field_name: str = 'name') -> None:
    """Validate that a name is valid for use as an environment variable.

    Raises:
        ValueError: If the name is invalid.
    """
    if not is_valid_env_var_name(name):
        raise ValueError(
            f"Invalid {field_name} '{name}'. Must start with a letter or underscore, "
            'and contain only alphanumeric characters and underscores.'
        )
