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
# Auto-enable enrichment if Azure DevOps credentials are configured
AZURE_DEVOPS_TENANT_ID = os.environ.get('AZURE_DEVOPS_TENANT_ID', '')
AZURE_DEVOPS_CLIENT_ID = os.environ.get('AZURE_DEVOPS_CLIENT_ID', '')
AZURE_DEVOPS_CLIENT_SECRET = os.environ.get('AZURE_DEVOPS_CLIENT_SECRET', '')
# Enable enrichment if all required credentials are present
AZURE_DEVOPS_ENABLED = bool(
    AZURE_DEVOPS_TENANT_ID and AZURE_DEVOPS_CLIENT_ID and AZURE_DEVOPS_CLIENT_SECRET
)

# Log enrichment status at module load
if AZURE_DEVOPS_ENABLED:
    logger.info('[AzureDevOpsEnrichment] Multi-org enrichment enabled')
else:
    logger.warning(
        '[AzureDevOpsEnrichment] Enrichment disabled. Missing configuration: '
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
        self, keycloak_user_id: str, email: str, organizations: list[str] | None = None
    ) -> bool:
        """Enrich a user's Keycloak profile with their Azure DevOps User IDs for all organizations.

        This method:
        1. Discovers all Azure DevOps organizations the user has access to (if not provided)
        2. For each organization, queries Azure DevOps Identities API to resolve email -> Azure DevOps ID
        3. Updates Keycloak user attributes with the Azure DevOps IDs as a JSON map {org: id}

        Args:
            keycloak_user_id: The Keycloak user ID
            email: User's email address (from Azure AD)
            organizations: Optional list of Azure DevOps organization names to enrich.
                          If None, discovers all organizations the user has access to.

        Returns:
            True if enrichment succeeded for at least one organization, False otherwise
        """
        if not AZURE_DEVOPS_ENABLED:
            return False

        if not email:
            logger.warning(
                f'[AzureDevOpsEnrichment] Cannot enrich user {keycloak_user_id}: '
                f'email is missing'
            )
            return False

        try:
            logger.info(
                f'[AzureDevOpsEnrichment] enrich_user_profile called for user {keycloak_user_id}, email={email}'
            )

            # Get Keycloak admin client
            keycloak_admin = get_keycloak_admin(self.external)

            # Get existing user attributes
            user = await keycloak_admin.a_get_user(keycloak_user_id)
            attributes = user.get('attributes', {})

            logger.info(
                f'[AzureDevOpsEnrichment] User attributes from Keycloak: {attributes}'
            )

            # Parse existing Azure DevOps IDs (stored as JSON map)
            import json

            existing_ids_map = {}
            if 'azure_devops_ids' in attributes:
                # Attributes are stored as lists in Keycloak
                ids_json_list = attributes['azure_devops_ids']
                if ids_json_list and len(ids_json_list) > 0:
                    try:
                        existing_ids_map = json.loads(ids_json_list[0])
                        logger.info(
                            f'[AzureDevOpsEnrichment] Existing azure_devops_ids map: {existing_ids_map}'
                        )
                    except json.JSONDecodeError:
                        logger.warning(
                            '[AzureDevOpsEnrichment] Failed to parse existing azure_devops_ids as JSON'
                        )

            # Discover organizations if not provided
            if not organizations:
                logger.info(
                    '[AzureDevOpsEnrichment] No organizations provided, discovering user organizations'
                )
                organizations = await self._discover_user_organizations(
                    keycloak_user_id
                )

            if not organizations:
                logger.warning(
                    '[AzureDevOpsEnrichment] No organizations found for user'
                )
                return False

            logger.info(
                f'[AzureDevOpsEnrichment] Enriching for {len(organizations)} organizations: {organizations}'
            )

            # Get service principal token for Azure DevOps API
            service_principal_token = await self._get_service_principal_token()
            if not service_principal_token:
                logger.error(
                    '[AzureDevOpsEnrichment] Failed to get service principal token'
                )
                return False

            # Enrich for each organization
            enriched_count = 0
            for org in organizations:
                # Skip if already enriched for this org
                if org in existing_ids_map:
                    logger.info(
                        f'[AzureDevOpsEnrichment] User already has Azure DevOps ID for org {org}, skipping'
                    )
                    enriched_count += 1
                    continue

                # Resolve email -> Azure DevOps ID for this organization
                azure_devops_id = await self._resolve_azure_devops_id(
                    email, org, service_principal_token
                )

                if azure_devops_id:
                    existing_ids_map[org] = azure_devops_id
                    enriched_count += 1
                    logger.info(
                        f'[AzureDevOpsEnrichment] Resolved Azure DevOps ID for {email} in org {org}: {azure_devops_id}'
                    )
                else:
                    logger.warning(
                        f'[AzureDevOpsEnrichment] Could not resolve Azure DevOps ID '
                        f'for email {email} in organization {org}'
                    )

            if enriched_count == 0:
                logger.warning(
                    '[AzureDevOpsEnrichment] Failed to enrich for any organization'
                )
                return False

            # Update Keycloak user attributes with the new map
            success = await self._update_keycloak_attributes(
                keycloak_user_id, existing_ids_map, user.get('email')
            )

            if not success:
                logger.error(
                    f'[AzureDevOpsEnrichment] Failed to update Keycloak attributes '
                    f'for user {keycloak_user_id}'
                )
                return False

            logger.info(
                f'[AzureDevOpsEnrichment] Successfully enriched {enriched_count}/{len(organizations)} organizations'
            )
            return True

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

    async def _discover_user_organizations(self, keycloak_user_id: str) -> list[str]:
        """Discover all Azure DevOps organizations the user has access to.

        Args:
            keycloak_user_id: The Keycloak user ID

        Returns:
            List of organization names, empty list if discovery fails
        """
        try:
            # Get user's access token from token manager
            from integrations.models import ProviderType

            access_token = await self.token_manager.get_access_token_from_user_id(
                keycloak_user_id, ProviderType.AZURE_DEVOPS
            )

            if not access_token:
                logger.warning(
                    f'[AzureDevOpsEnrichment] No Azure DevOps access token found for user {keycloak_user_id}'
                )
                return []

            # Query Azure DevOps for user's organizations
            import httpx

            # Get user profile to find member ID
            profile_url = 'https://app.vssps.visualstudio.com/_apis/profile/profiles/me?api-version=7.1-preview.3'
            headers = {
                'Authorization': f'Bearer {access_token.get_secret_value()}',
                'Content-Type': 'application/json',
            }

            async with httpx.AsyncClient() as client:
                profile_response = await client.get(profile_url, headers=headers)
                profile_response.raise_for_status()
                profile_data = profile_response.json()
                member_id = profile_data.get('id')

                if not member_id:
                    logger.error(
                        '[AzureDevOpsEnrichment] Failed to get member ID from profile'
                    )
                    return []

                # Get organizations for this member
                accounts_url = f'https://app.vssps.visualstudio.com/_apis/accounts?memberId={member_id}&api-version=7.1'
                accounts_response = await client.get(accounts_url, headers=headers)
                accounts_response.raise_for_status()

                accounts_data = accounts_response.json()
                organizations = accounts_data.get('value', [])

                org_names = [
                    org['accountName']
                    for org in organizations
                    if 'accountName' in org and org['accountName']
                ]

                logger.info(
                    f'[AzureDevOpsEnrichment] Discovered {len(org_names)} organizations: {org_names}'
                )
                return org_names

        except Exception as e:
            logger.error(
                f'[AzureDevOpsEnrichment] Failed to discover organizations: {e}',
                exc_info=True,
            )
            return []

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

    async def _update_keycloak_attributes(
        self, keycloak_user_id: str, azure_devops_ids_map: dict[str, str], email: str
    ) -> bool:
        """Update Keycloak user with Azure DevOps IDs map attribute.

        Args:
            keycloak_user_id: The Keycloak user ID
            azure_devops_ids_map: Map of organization -> Azure DevOps User ID
            email: User's email (required by Keycloak)

        Returns:
            True if update succeeded, False otherwise
        """
        try:
            import json

            keycloak_admin = get_keycloak_admin(self.external)

            # Get existing user attributes first to merge with new attribute
            user = await keycloak_admin.a_get_user(keycloak_user_id)
            existing_attributes = user.get('attributes', {})

            logger.debug(
                f'[AzureDevOpsEnrichment] User {keycloak_user_id} existing attributes: {existing_attributes}'
            )

            # Store the IDs map as JSON string (Keycloak stores as lists)
            ids_json = json.dumps(azure_devops_ids_map)
            existing_attributes['azure_devops_ids'] = [ids_json]

            logger.debug(
                f'[AzureDevOpsEnrichment] Updated attributes payload: {existing_attributes}'
            )

            # Include email at top level to satisfy Keycloak's required field validation
            payload = {'attributes': existing_attributes, 'email': email}

            await keycloak_admin.a_update_user(keycloak_user_id, payload)
            logger.info(
                f'[AzureDevOpsEnrichment] Successfully updated azure_devops_ids for user {keycloak_user_id}: {azure_devops_ids_map}'
            )
            return True

        except Exception as e:
            logger.error(
                f'[AzureDevOpsEnrichment] Failed to update Keycloak attributes: {e}',
                exc_info=True,
            )
            return False


def schedule_azure_devops_enrichment(user_id: str, email: str) -> None:
    """Schedule Azure DevOps user profile enrichment as a background task.

    This function should be called after user login to enrich their profile
    with Azure DevOps User IDs for all organizations they have access to.

    Args:
        user_id: Keycloak user ID
        email: User's email address
    """
    if not AZURE_DEVOPS_ENABLED:
        logger.warning(
            '[AzureDevOpsEnrichment] Azure DevOps enrichment disabled. '
            'Set AZURE_DEVOPS_ENABLED=true to enable user ID mapping.'
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
            result = await enricher.enrich_user_profile(user_id, email)
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
