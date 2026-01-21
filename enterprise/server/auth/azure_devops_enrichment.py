"""Azure DevOps user profile enrichment for Keycloak.

This module enriches Keycloak user profiles with Azure DevOps User IDs during login.
This enables proper user mapping when Azure DevOps webhooks arrive with Azure DevOps IDs
that differ from the Azure AD Object IDs stored in Keycloak.
"""

import asyncio
import os
from typing import Optional

from pydantic import SecretStr
from server.auth.keycloak_manager import get_keycloak_admin
from server.auth.token_manager import TokenManager

from openhands.core.logger import openhands_logger as logger

# Azure DevOps configuration
# Auto-enable enrichment if Azure DevOps organization and credentials are configured
AZURE_DEVOPS_ORGANIZATION = os.environ.get('AZURE_DEVOPS_ORGANIZATION', '')
AZURE_DEVOPS_TENANT_ID = os.environ.get('AZURE_DEVOPS_TENANT_ID', '')
AZURE_DEVOPS_CLIENT_ID = os.environ.get('AZURE_DEVOPS_CLIENT_ID', '')
AZURE_DEVOPS_CLIENT_SECRET = os.environ.get('AZURE_DEVOPS_CLIENT_SECRET', '')
# Enable enrichment if all required configuration is present
AZURE_DEVOPS_ENABLED = bool(
    AZURE_DEVOPS_ORGANIZATION
    and AZURE_DEVOPS_TENANT_ID
    and AZURE_DEVOPS_CLIENT_ID
    and AZURE_DEVOPS_CLIENT_SECRET
)

# Log enrichment status at module load
if AZURE_DEVOPS_ENABLED:
    logger.info(
        f'[AzureDevOpsEnrichment] Enrichment enabled for organization: {AZURE_DEVOPS_ORGANIZATION}'
    )
else:
    logger.warning(
        '[AzureDevOpsEnrichment] Enrichment disabled. Missing configuration: '
        f'organization={bool(AZURE_DEVOPS_ORGANIZATION)}, '
        f'tenant_id={bool(AZURE_DEVOPS_TENANT_ID)}, '
        f'client_id={bool(AZURE_DEVOPS_CLIENT_ID)}, '
        f'client_secret={bool(AZURE_DEVOPS_CLIENT_SECRET)}'
    )


class AzureDevOpsUserEnricher:
    """Enriches Keycloak user profiles with Azure DevOps User IDs."""

    def __init__(self, external_token_manager: bool = False):
        """Initialize the enricher.

        Args:
            external_token_manager: Whether to use external token manager
        """
        self.token_manager = TokenManager(external=external_token_manager)
        self.external = external_token_manager

    async def enrich_user_profile(
        self, keycloak_user_id: str, email: str, organization: Optional[str] = None
    ) -> bool:
        """Enrich a user's Keycloak profile with their Azure DevOps User ID.

        This method:
        1. Checks if user already has azure_devops_id attribute
        2. If not, queries Azure DevOps Graph API to resolve email -> Azure DevOps ID
        3. Updates Keycloak user attributes with the Azure DevOps ID

        Args:
            keycloak_user_id: The Keycloak user ID
            email: User's email address (from Azure AD)
            organization: Azure DevOps organization name (defaults to env var)

        Returns:
            True if enrichment succeeded, False otherwise
        """
        if not AZURE_DEVOPS_ENABLED:
            return False

        if not email:
            logger.warning(
                f'[AzureDevOpsEnrichment] Cannot enrich user {keycloak_user_id}: '
                f'email is missing'
            )
            return False

        org = organization or AZURE_DEVOPS_ORGANIZATION
        if not org:
            logger.warning(
                '[AzureDevOpsEnrichment] Cannot enrich user: '
                'Azure DevOps organization not configured'
            )
            return False

        try:
            logger.info(
                f'[AzureDevOpsEnrichment] enrich_user_profile called for user {keycloak_user_id}, email={email}, org={org}'
            )

            # Get Keycloak admin client
            keycloak_admin = get_keycloak_admin(self.external)

            # Check if user already has azure_devops_id
            user = await keycloak_admin.a_get_user(keycloak_user_id)
            attributes = user.get('attributes', {})

            logger.info(
                f'[AzureDevOpsEnrichment] User attributes from Keycloak: {attributes}'
            )

            existing_azure_devops_id = None
            if 'azure_devops_id' in attributes:
                # Attributes are stored as lists in Keycloak
                azure_devops_ids = attributes['azure_devops_id']
                if azure_devops_ids and len(azure_devops_ids) > 0:
                    existing_azure_devops_id = azure_devops_ids[0]

            logger.info(
                f'[AzureDevOpsEnrichment] Existing azure_devops_id: {existing_azure_devops_id}'
            )

            if existing_azure_devops_id:
                logger.info(
                    '[AzureDevOpsEnrichment] User already has azure_devops_id, skipping enrichment'
                )
                return True

            logger.info(
                '[AzureDevOpsEnrichment] No existing azure_devops_id found, proceeding with enrichment'
            )

            # Get service principal token for Azure DevOps API
            service_principal_token = await self._get_service_principal_token()
            if not service_principal_token:
                logger.error(
                    '[AzureDevOpsEnrichment] Failed to get service principal token'
                )
                return False

            # Resolve email -> Azure DevOps ID using Graph API
            azure_devops_id = await self._resolve_azure_devops_id(
                email, org, service_principal_token
            )

            if not azure_devops_id:
                logger.warning(
                    f'[AzureDevOpsEnrichment] Could not resolve Azure DevOps ID '
                    f'for email {email} in organization {org}. User will not be mapped to webhooks.'
                )
                return False

            # Update Keycloak user attributes
            success = await self._update_keycloak_attribute(
                keycloak_user_id, azure_devops_id
            )

            if not success:
                logger.error(
                    f'[AzureDevOpsEnrichment] Failed to update Keycloak attributes '
                    f'for user {keycloak_user_id}'
                )

            return success

        except Exception as e:
            logger.error(
                f'[AzureDevOpsEnrichment] Error enriching user {keycloak_user_id}: {e}',
                exc_info=True,
            )
            return False

    async def _get_service_principal_token(self) -> Optional[SecretStr]:
        """Get service principal token for Azure DevOps Graph API access.

        Returns:
            Service principal access token or None if unavailable
        """
        if not all(
            [
                AZURE_DEVOPS_TENANT_ID,
                AZURE_DEVOPS_CLIENT_ID,
                AZURE_DEVOPS_CLIENT_SECRET,
            ]
        ):
            logger.warning(
                '[AzureDevOpsEnrichment] Service principal credentials not configured'
            )
            return None

        try:
            import httpx

            # Get token using client credentials flow
            token_url = f'https://login.microsoftonline.com/{AZURE_DEVOPS_TENANT_ID}/oauth2/v2.0/token'
            data = {
                'client_id': AZURE_DEVOPS_CLIENT_ID,
                'scope': 'https://app.vssps.visualstudio.com/.default',
                'client_secret': AZURE_DEVOPS_CLIENT_SECRET,
                'grant_type': 'client_credentials',
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(token_url, data=data)
                response.raise_for_status()
                token_data = response.json()
                access_token = token_data.get('access_token')

                if access_token:
                    return SecretStr(access_token)
                else:
                    logger.error('[AzureDevOpsEnrichment] No access_token in response')
                    return None

        except Exception as e:
            logger.error(
                f'[AzureDevOpsEnrichment] Failed to get service principal token: {e}',
                exc_info=True,
            )
            return None

    async def _resolve_azure_devops_id(
        self, email: str, organization: str, token: SecretStr
    ) -> Optional[str]:
        """Resolve user email to Azure DevOps User ID using Identities API.

        Args:
            email: User's email address
            organization: Azure DevOps organization name
            token: Service principal access token

        Returns:
            Azure DevOps User ID (VSID/Storage Key) that matches webhook payloads,
            or None if not found
        """
        try:
            # Import the resolver (lazy import to avoid circular dependencies)
            from integrations.azure_devops.azure_devops_id_resolver import (
                AzureDevOpsIdResolver,
            )

            resolver = AzureDevOpsIdResolver(token)
            azure_devops_id = await resolver.get_azure_devops_id_from_email(
                email, organization
            )

            return azure_devops_id

        except Exception as e:
            logger.error(
                f'[AzureDevOpsEnrichment] Error resolving Azure DevOps ID: {e}',
                exc_info=True,
            )
            return None

    async def _update_keycloak_attribute(
        self, keycloak_user_id: str, azure_devops_id: str
    ) -> bool:
        """Update Keycloak user with Azure DevOps ID attribute.

        Args:
            keycloak_user_id: The Keycloak user ID
            azure_devops_id: The Azure DevOps User ID to store

        Returns:
            True if update succeeded, False otherwise
        """
        try:
            keycloak_admin = get_keycloak_admin(self.external)

            # Get existing user attributes first to merge with new attribute
            user = await keycloak_admin.a_get_user(keycloak_user_id)
            existing_attributes = user.get('attributes', {})

            logger.debug(
                f'[AzureDevOpsEnrichment] User {keycloak_user_id} existing attributes: {existing_attributes}'
            )
            logger.debug(
                f'[AzureDevOpsEnrichment] User email from top-level: {user.get("email")}'
            )

            # Merge azure_devops_id with existing attributes (Keycloak stores as lists)
            existing_attributes['azure_devops_id'] = [azure_devops_id]

            logger.debug(
                f'[AzureDevOpsEnrichment] Updated attributes payload: {existing_attributes}'
            )

            # Include email at top level to satisfy Keycloak's required field validation
            payload = {'attributes': existing_attributes, 'email': user.get('email')}

            await keycloak_admin.a_update_user(keycloak_user_id, payload)
            logger.info(
                f'[AzureDevOpsEnrichment] Successfully updated azure_devops_id for user {keycloak_user_id}'
            )
            return True

        except Exception as e:
            logger.error(
                f'[AzureDevOpsEnrichment] Failed to update Keycloak attribute: {e}',
                exc_info=True,
            )
            return False


def schedule_azure_devops_enrichment(
    user_id: str, email: str, organization: Optional[str] = None
) -> None:
    """Schedule Azure DevOps user profile enrichment as a background task.

    This function should be called after user login to enrich their profile
    with Azure DevOps User ID without blocking the login flow.

    Args:
        user_id: Keycloak user ID
        email: User's email address
        organization: Azure DevOps organization (optional, defaults to env var)
    """
    if not AZURE_DEVOPS_ENABLED:
        logger.warning(
            '[AzureDevOpsEnrichment] Azure DevOps enrichment disabled. '
            'Set AZURE_DEVOPS_ENABLED=true to enable user ID mapping.'
        )
        return

    if not AZURE_DEVOPS_ORGANIZATION:
        logger.error(
            f'[AzureDevOpsEnrichment] AZURE_DEVOPS_ORGANIZATION not configured. '
            f'Cannot enrich user {user_id}.'
        )
        return

    if not all(
        [AZURE_DEVOPS_TENANT_ID, AZURE_DEVOPS_CLIENT_ID, AZURE_DEVOPS_CLIENT_SECRET]
    ):
        logger.error(
            f'[AzureDevOpsEnrichment] Service principal credentials not configured. '
            f'Cannot enrich user {user_id}.'
        )
        return

    logger.info(
        f'[AzureDevOpsEnrichment] Scheduling enrichment for user {user_id} (email: {email})'
    )

    async def _enrich():
        """Background enrichment task."""
        try:
            logger.info(
                f'[AzureDevOpsEnrichment] Starting enrichment for user {user_id} (email: {email})'
            )
            enricher = AzureDevOpsUserEnricher(external_token_manager=True)
            result = await enricher.enrich_user_profile(user_id, email, organization)
            if result:
                logger.info(
                    f'[AzureDevOpsEnrichment] Successfully enriched user {user_id}'
                )
            else:
                logger.warning(
                    f'[AzureDevOpsEnrichment] Enrichment returned False for user {user_id}'
                )
        except Exception as e:
            logger.error(
                f'[AzureDevOpsEnrichment] Background enrichment failed: {e}',
                exc_info=True,
            )

    # Schedule as background task
    asyncio.create_task(_enrich())
