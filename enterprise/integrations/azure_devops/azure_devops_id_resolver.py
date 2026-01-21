"""Azure DevOps ID Resolver - Maps email addresses to Azure DevOps User IDs.

This service queries the Azure DevOps Identities API to resolve a user's email address
to their Azure DevOps User ID (VSID/Storage Key). This ID is what webhooks send in the
revisedBy.id field.

IMPORTANT: We use the Identities API (not Graph API) because:
- Identities API returns 'id' field = VSID/Storage Key (matches webhook payloads)
- Graph API returns 'originId' field = Azure AD Object ID (does NOT match webhooks)

The resolver implements caching using TTLCache to minimize API calls and improve performance.
"""

from typing import Optional

import httpx
from cachetools import TTLCache  # type: ignore[import-untyped]
from pydantic import SecretStr

from openhands.core.logger import openhands_logger as logger

# Cache configuration
CACHE_TTL_SECONDS = 3600  # 1 hour
MAX_CACHE_SIZE = 10000  # Per cache instance


class AzureDevOpsIdResolver:
    """Resolves user email addresses to Azure DevOps User IDs using the Identities API."""

    def __init__(self, token: SecretStr):
        """Initialize the resolver with an Azure DevOps access token.

        Args:
            token: Azure DevOps service principal or PAT token with vso.graph scope
        """
        self.token = token
        # Forward cache: (org, email) -> azure_devops_id
        self._user_cache: TTLCache = TTLCache(
            maxsize=MAX_CACHE_SIZE, ttl=CACHE_TTL_SECONDS
        )
        # Reverse cache: (org, azure_devops_id) -> email
        self._reverse_cache: TTLCache = TTLCache(
            maxsize=MAX_CACHE_SIZE, ttl=CACHE_TTL_SECONDS
        )

    async def get_azure_devops_id_from_email(
        self, email: str, organization: str
    ) -> Optional[str]:
        """Get Azure DevOps User ID from email address.

        Args:
            email: User's email address
            organization: Azure DevOps organization name (e.g., "AIngineer")

        Returns:
            Azure DevOps User ID (VSID/Storage Key) that matches webhook payloads,
            or None if not found
        """
        if not email or not organization:
            logger.warning(
                f'[AzureDevOpsIdResolver] Missing required parameters: '
                f'email={email}, organization={organization}'
            )
            return None

        email_lower = email.lower()
        cache_key = (organization, email_lower)

        # Check cache first
        cached_id = self._user_cache.get(cache_key)
        if cached_id is not None:
            logger.debug(f'[AzureDevOpsIdResolver] Cache hit for {email}')
            return cached_id

        # Cache miss - query Identities API
        logger.info(f'[AzureDevOpsIdResolver] Querying Identities API for {email}')
        azure_devops_id = await self._fetch_user_id_by_email(email_lower, organization)

        if azure_devops_id:
            # Update both forward and reverse caches
            self._cache_user(organization, email_lower, azure_devops_id)
            logger.info(
                f'[AzureDevOpsIdResolver] Found Azure DevOps ID for {email}: {azure_devops_id}'
            )
        else:
            logger.warning(f'[AzureDevOpsIdResolver] User not found: {email}')

        return azure_devops_id

    async def get_user_email_from_id(
        self, azure_devops_id: str, organization: str
    ) -> Optional[str]:
        """Get user email from Azure DevOps User ID (reverse lookup).

        This is the Plan B fallback: when we receive a webhook with an Azure DevOps ID
        but don't have it stored in Keycloak, we can look up the email.

        Args:
            azure_devops_id: Azure DevOps User ID (VSID/Storage Key from webhook)
            organization: Azure DevOps organization name

        Returns:
            User's email address or None if not found
        """
        if not azure_devops_id or not organization:
            logger.warning(
                f'[AzureDevOpsIdResolver] Missing required parameters: '
                f'azure_devops_id={azure_devops_id}, organization={organization}'
            )
            return None

        azure_devops_id_lower = azure_devops_id.lower()
        cache_key = (organization, azure_devops_id_lower)

        # Check reverse cache first
        cached_email = self._reverse_cache.get(cache_key)
        if cached_email is not None:
            logger.debug(
                f'[AzureDevOpsIdResolver] Reverse cache hit for {azure_devops_id}'
            )
            return cached_email

        # Cache miss - query Identities API
        logger.info(
            f'[AzureDevOpsIdResolver] Querying Identities API for ID {azure_devops_id}'
        )
        email = await self._fetch_email_by_id(azure_devops_id, organization)

        if email:
            # Update both forward and reverse caches
            self._cache_user(organization, email, azure_devops_id)
            logger.info(
                f'[AzureDevOpsIdResolver] Found email for Azure DevOps ID {azure_devops_id}: {email}'
            )
        else:
            logger.warning(
                f'[AzureDevOpsIdResolver] Azure DevOps ID not found: {azure_devops_id}'
            )

        return email

    def _cache_user(self, organization: str, email: str, azure_devops_id: str) -> None:
        """Cache a user's email <-> Azure DevOps ID mapping.

        Args:
            organization: Azure DevOps organization name
            email: User's email address (lowercase)
            azure_devops_id: Azure DevOps User ID (VSID/Storage Key)
        """
        email_lower = email.lower()
        azure_devops_id_lower = azure_devops_id.lower()

        # Store forward lookup (email -> id)
        self._user_cache[(organization, email_lower)] = azure_devops_id

        # Store reverse lookup (id -> email)
        self._reverse_cache[(organization, azure_devops_id_lower)] = email_lower

    async def _fetch_user_id_by_email(
        self, email: str, organization: str
    ) -> Optional[str]:
        """Query Identities API to get Azure DevOps User ID for an email.

        Args:
            email: User's email address
            organization: Azure DevOps organization name

        Returns:
            Azure DevOps User ID (VSID/Storage Key) or None if not found
        """
        try:
            headers = {
                'Authorization': f'Bearer {self.token.get_secret_value()}',
                'Content-Type': 'application/json',
            }

            # Query Identities API with email as search filter
            url = f'https://vssps.dev.azure.com/{organization}/_apis/identities'
            params = {
                'api-version': '7.1',
                'searchFilter': 'General',
                'filterValue': email,
                'queryMembership': 'None',
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()

                data = response.json()
                identities = data.get('value', [])

                # Find matching identity
                for identity in identities:
                    # Extract email from properties
                    properties = identity.get('properties', {})
                    identity_email = None

                    if 'Account' in properties:
                        account = properties['Account']
                        if isinstance(account, dict):
                            identity_email = account.get('$value', '').lower()

                    # Also check providerDisplayName
                    if not identity_email:
                        identity_email = identity.get('providerDisplayName', '').lower()
                        if identity_email and '@' not in identity_email:
                            identity_email = None

                    # Match email and return ID
                    if identity_email and identity_email == email.lower():
                        azure_devops_id = identity.get('id')
                        if azure_devops_id:
                            logger.info(
                                f'[AzureDevOpsIdResolver] Found identity for {email}: '
                                f'id={azure_devops_id}, descriptor={identity.get("descriptor")}'
                            )
                            return azure_devops_id

                logger.warning(
                    f'[AzureDevOpsIdResolver] No matching identity found for {email}'
                )
                return None

        except httpx.HTTPStatusError as e:
            logger.error(
                f'[AzureDevOpsIdResolver] HTTP error querying identity for {email}: '
                f'{e.response.status_code} - {e.response.text}'
            )
            return None
        except Exception as e:
            logger.error(
                f'[AzureDevOpsIdResolver] Error querying identity for {email}: {e}',
                exc_info=True,
            )
            return None

    async def _fetch_email_by_id(
        self, azure_devops_id: str, organization: str
    ) -> Optional[str]:
        """Query Identities API to get email for an Azure DevOps User ID.

        Args:
            azure_devops_id: Azure DevOps User ID (VSID/Storage Key)
            organization: Azure DevOps organization name

        Returns:
            User's email address or None if not found
        """
        try:
            headers = {
                'Authorization': f'Bearer {self.token.get_secret_value()}',
                'Content-Type': 'application/json',
            }

            # Query specific identity by ID
            url = f'https://vssps.dev.azure.com/{organization}/_apis/identities'
            params = {
                'api-version': '7.1',
                'identityIds': azure_devops_id,
                'queryMembership': 'None',
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()

                data = response.json()
                identities = data.get('value', [])

                if identities:
                    identity = identities[0]
                    properties = identity.get('properties', {})

                    # Extract email
                    email = None
                    if 'Account' in properties:
                        account = properties['Account']
                        if isinstance(account, dict):
                            email = account.get('$value', '').lower()

                    if not email:
                        email = identity.get('providerDisplayName', '').lower()
                        if email and '@' not in email:
                            email = None

                    if email:
                        logger.info(
                            f'[AzureDevOpsIdResolver] Found email for ID {azure_devops_id}: {email}'
                        )
                        return email

                logger.warning(
                    f'[AzureDevOpsIdResolver] No identity found for ID {azure_devops_id}'
                )
                return None

        except httpx.HTTPStatusError as e:
            logger.error(
                f'[AzureDevOpsIdResolver] HTTP error querying identity for ID {azure_devops_id}: '
                f'{e.response.status_code} - {e.response.text}'
            )
            return None
        except Exception as e:
            logger.error(
                f'[AzureDevOpsIdResolver] Error querying identity for ID {azure_devops_id}: {e}',
                exc_info=True,
            )
            return None

    def invalidate_cache(self, organization: Optional[str] = None) -> None:
        """Manually invalidate cache for an organization or all organizations.

        Args:
            organization: Specific organization to invalidate, or None for all
        """
        if organization:
            # Remove entries for specific organization
            keys_to_remove = [k for k in self._user_cache if k[0] == organization]
            for key in keys_to_remove:
                del self._user_cache[key]

            keys_to_remove = [k for k in self._reverse_cache if k[0] == organization]
            for key in keys_to_remove:
                del self._reverse_cache[key]

            logger.info(
                f'[AzureDevOpsIdResolver] Invalidated cache for organization: {organization}'
            )
        else:
            self._user_cache.clear()
            self._reverse_cache.clear()
            logger.info('[AzureDevOpsIdResolver] Invalidated all caches')
