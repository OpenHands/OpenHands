"""Credential resolver for matching resources to stored credentials."""

from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from openhands.core.logger import openhands_logger as logger
from openhands.storage.data_models.credential_mapping import CredentialMapping

if TYPE_CHECKING:
    from openhands.storage.data_models.secrets import Secrets


class CredentialResolver:
    """Resolves credentials for resources based on URL patterns.

    This class matches resource URLs (from HTTP requests, npm, docker, etc.)
    to stored credential mappings and returns the appropriate authentication
    headers or credentials.
    """

    def __init__(self, secrets: Secrets | None = None):
        """Initialize the credential resolver with secrets.

        Args:
            secrets: The Secrets object containing credential mappings and custom secrets
        """
        self.secrets = secrets
        self._mappings_cache: dict[str, CredentialMapping] | None = None

    def _get_mappings(self) -> dict[str, CredentialMapping]:
        """Get credential mappings, using cache if available."""
        if self._mappings_cache is None:
            if self.secrets and self.secrets.credential_mappings:
                self._mappings_cache = dict(self.secrets.credential_mappings)
            else:
                self._mappings_cache = {}
        return self._mappings_cache

    def resolve_credential(self, url: str) -> tuple[str, dict[str, str]] | None:
        """Resolve credentials for a given URL.

        This method matches the URL against stored credential mappings and
        returns the credential value along with the headers needed for authentication.

        Args:
            url: The URL of the resource (e.g., "https://npm.example.com/package",
                "npm.example.com", "https://api.example.com/v1/resource")

        Returns:
            A tuple of (credential_value, auth_headers) if a match is found,
            None otherwise. auth_headers is a dictionary of header name to value.
        """
        if not self.secrets or not self.secrets.credential_mappings:
            return None

        # Try to find a matching credential mapping
        mapping = self._match_url_to_mapping(url)
        if not mapping:
            return None

        # Get the credential value from custom secrets
        if mapping.credential_name not in self.secrets.custom_secrets:
            logger.warning(
                f'Credential mapping references non-existent secret: {mapping.credential_name}'
            )
            return None

        credential_secret = self.secrets.custom_secrets[mapping.credential_name]
        credential_value = credential_secret.secret.get_secret_value()

        # Build authentication headers based on auth_method
        auth_headers = self._build_auth_headers(mapping, credential_value)

        return credential_value, auth_headers

    def _match_url_to_mapping(self, url: str) -> CredentialMapping | None:
        """Match a URL to a credential mapping pattern.

        Matching priority:
        1. Exact URL match
        2. Domain match (with or without protocol)
        3. Wildcard pattern match

        Args:
            url: The URL to match

        Returns:
            The matching CredentialMapping if found, None otherwise
        """
        mappings = self._get_mappings()

        if not mappings:
            return None

        # Parse URL to extract domain
        parsed = self._parse_url(url)
        domain = parsed.get('domain', '')
        full_url = parsed.get('full_url', url)

        # Try exact match first
        for mapping_id, mapping in mappings.items():
            if mapping.resource_pattern == url or mapping.resource_pattern == full_url:
                logger.debug(f'Exact match found for URL {url}: mapping {mapping_id}')
                return mapping

        # Try domain match
        for mapping_id, mapping in mappings.items():
            pattern = mapping.resource_pattern
            # Remove protocol if present in pattern
            if pattern.startswith(('http://', 'https://')):
                pattern_parsed = urlparse(pattern)
                pattern_domain = (
                    pattern_parsed.netloc or pattern_parsed.path.split('/')[0]
                )
            else:
                pattern_domain = pattern.split('/')[0]

            if domain and domain == pattern_domain:
                logger.debug(f'Domain match found for URL {url}: mapping {mapping_id}')
                return mapping

        # Try wildcard/fnmatch pattern
        for mapping_id, mapping in mappings.items():
            pattern = mapping.resource_pattern
            # Convert wildcard pattern to regex-like matching
            if self._pattern_matches(pattern, url) or self._pattern_matches(
                pattern, domain
            ):
                logger.debug(f'Pattern match found for URL {url}: mapping {mapping_id}')
                return mapping

        return None

    def _parse_url(self, url: str) -> dict[str, str]:
        """Parse a URL to extract domain and full URL.

        Handles both full URLs (with protocol) and domain-only URLs.

        Args:
            url: The URL to parse

        Returns:
            Dictionary with 'domain' and 'full_url' keys
        """
        if not url:
            return {'domain': '', 'full_url': url}

        # If URL doesn't start with a protocol, try to parse as domain
        if not url.startswith(('http://', 'https://')):
            # Assume it's a domain or domain/path
            domain = url.split('/')[0]
            return {'domain': domain, 'full_url': f'https://{url}'}

        # Parse full URL
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split('/')[0]

        return {'domain': domain, 'full_url': url}

    def _pattern_matches(self, pattern: str, text: str) -> bool:
        """Check if a pattern matches text using fnmatch.

        Supports wildcards like * and ?.

        Args:
            pattern: The pattern to match (may contain wildcards)
            text: The text to match against

        Returns:
            True if pattern matches text, False otherwise
        """
        try:
            return fnmatch.fnmatch(text, pattern) or fnmatch.fnmatch(
                text.lower(), pattern.lower()
            )
        except Exception:
            # If fnmatch fails, try simple substring match
            return pattern.lower() in text.lower()

    def _build_auth_headers(
        self, mapping: CredentialMapping, credential_value: str
    ) -> dict[str, str]:
        """Build authentication headers based on the auth_method.

        Args:
            mapping: The credential mapping containing auth configuration
            credential_value: The actual credential value (token, API key, etc.)

        Returns:
            Dictionary of header name to header value
        """
        headers: dict[str, str] = {}

        if mapping.auth_method == 'bearer_token':
            headers['Authorization'] = f'Bearer {credential_value}'
        elif mapping.auth_method == 'api_key':
            # API key can be used in different ways depending on the service
            # Default to X-API-Key header, but can be customized via auth_header
            header_name = mapping.auth_header or 'X-API-Key'
            headers[header_name] = credential_value
        elif mapping.auth_method == 'basic_auth':
            # Basic auth expects base64-encoded username:password
            # For simplicity, assume credential_value is already in the right format
            # or is a token that should be used as password with empty username
            import base64

            # Try to decode to see if it's already base64, otherwise encode username:password
            try:
                base64.b64decode(credential_value)
                # Already base64 encoded
                headers['Authorization'] = f'Basic {credential_value}'
            except Exception:
                # Not base64, assume it's a token/password
                # Use empty username or token as both username and password
                auth_string = f':{credential_value}'
                encoded = base64.b64encode(auth_string.encode()).decode()
                headers['Authorization'] = f'Basic {encoded}'
        elif mapping.auth_method == 'header':
            if not mapping.auth_header:
                raise ValueError(
                    'auth_header must be specified when auth_method is "header"'
                )
            headers[mapping.auth_header] = credential_value
        else:
            # This else is kept for defensive programming in case auth_method enum is extended
            logger.warning(  # type: ignore[unreachable]
                f'Unknown auth_method: {mapping.auth_method} for mapping {mapping.resource_pattern}'
            )

        return headers

    def list_mappings(self) -> list[CredentialMapping]:
        """List all credential mappings.

        Returns:
            List of all CredentialMapping objects
        """
        mappings = self._get_mappings()
        return list(mappings.values())

    def get_mapping(self, mapping_id: str) -> CredentialMapping | None:
        """Get a specific credential mapping by ID.

        Args:
            mapping_id: The ID of the mapping

        Returns:
            The CredentialMapping if found, None otherwise
        """
        mappings = self._get_mappings()
        return mappings.get(mapping_id)
