"""BitbucketManager handles incoming webhook events from Bitbucket."""

import hashlib
import os
from typing import Optional

import aiohttp
from integrations.models import Message, SourceType
from integrations.types import UserData
from server.auth.token_manager import TokenManager

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.provider import ProviderType


class BitbucketManager:
    """Manages Bitbucket webhook events and PR comment processing."""

    def __init__(self, token_manager: Optional[TokenManager] = None):
        self.token_manager = token_manager or TokenManager()
        logger.info('BitbucketManager initialized')

    def _extract_user_info(self, event_data: dict) -> Optional[UserData]:
        actor = event_data.get('actor', {})
        if not actor:
            return None

        account_id = actor.get('accountId', '')
        username = actor.get('username', '')

        if not account_id:
            return None

        user_id_int = int(hashlib.sha256(account_id.encode()).hexdigest()[:8], 16)

        return UserData(
            user_id=user_id_int,
            username=username or account_id,
            keycloak_user_id=None,
            bitbucket_id=account_id,
        )

    async def receive_message(self, message: Message):
        if message.source != SourceType.BITBUCKET:
            logger.error(f'Received non-Bitbucket message: {message.source}')
            return

        payload = message.message
        if not isinstance(payload, dict):
            logger.warning(f'Received unexpected message format: {type(payload)}')
            return

        event_data = payload.get('event', {})

        user_info = self._extract_user_info(event_data)

        workspace_slug = event_data.get('workspace', {}).get('slug')
        repo_slug = event_data.get('repository', {}).get('slug')
        pr_data = event_data.get('pullrequest', {})
        pr_id = pr_data.get('id')
        comment_text = event_data.get('comment', {}).get('content', {}).get('raw', '')

        logger.info(
            f'BitbucketManager received PR comment:\n'
            f'  Workspace: {workspace_slug}\n'
            f'  Repository: {repo_slug}\n'
            f'  PR: #{pr_id}\n'
            f'  User: {user_info.username if user_info else "Unknown"} (ID: {user_info.user_id if user_info else "Unknown"})\n'
            f'  Comment: {comment_text[:100]}...'
        )

        if '@openhands' not in comment_text.lower():
            logger.debug('Comment does not mention @openhands, ignoring')
            return

        user_found = False
        if user_info:
            bitbucket_account_id = user_info.bitbucket_id
            keycloak_user_id = await self.token_manager.get_user_id_from_idp_user_id(
                bitbucket_account_id, ProviderType.BITBUCKET
            )
            user_found = keycloak_user_id is not None
            if user_found:
                user_info.keycloak_user_id = keycloak_user_id

        acknowledgment_message = (
            "🤖 **OpenHands** received your request!\n\n"
            "📋 **Details received:**\n\n"
            f"- Workspace: `{workspace_slug}`\n\n"
            f"- Repository: `{repo_slug}`\n\n"
            f"- PR: #{pr_id}\n\n"
            f"- User: {user_info.username if user_info else 'Unknown'}\n\n"
            f"- User found in Keycloak: {'✅ Yes' if user_found else '❌ No'}"
        )

        if workspace_slug and repo_slug and pr_id:
            await self.send_response_to_pr(
                workspace_slug, repo_slug, pr_id, acknowledgment_message
            )

        if user_found:
            await self._check_user_authorization(
                user_info, workspace_slug, repo_slug, pr_id
            )
        else:
            if user_info:
                logger.warning(f'User {user_info.bitbucket_id} not found in Keycloak')
            else:
                logger.warning('No user info extracted from webhook')

    async def _check_user_authorization(
        self,
        user_info: UserData,
        workspace_slug: str,
        repo_slug: str,
        pr_id: Optional[int],
    ):
        bitbucket_account_id = user_info.bitbucket_id

        user_token = await self.token_manager.get_idp_token_from_idp_user_id(
            bitbucket_account_id, ProviderType.BITBUCKET
        )

        if user_token:
            logger.info(f'Found Bitbucket token for user {bitbucket_account_id}')

            has_write_access = await self._check_write_access(
                user_token, workspace_slug, repo_slug, user_info
            )

            if has_write_access:
                logger.info(
                    f'User {bitbucket_account_id} has write access to {workspace_slug}/{repo_slug}'
                )
                # TODO: Trigger OpenHands agent with PR context
            else:
                logger.warning(
                    f'User {bitbucket_account_id} does not have write access to {workspace_slug}/{repo_slug}'
                )
        else:
            logger.warning(
                f'No Bitbucket token found for user {bitbucket_account_id}. '
                f'User needs to re-authenticate with Bitbucket.'
            )

    async def _check_write_access(
        self, user_token: str, workspace_slug: str, repo_slug: str, user_info: UserData
    ) -> bool:
        # TODO: Implement actual permission check using Bitbucket API
        # API endpoint: GET /repositories/{workspace}/{repo_slug}/permissions-config/users/{selected_user_id}
        # This should be implemented after Keycloak Bitbucket IDP support is merged

        logger.info(
            f'Checking write access for user {user_info.bitbucket_id} '
            f'to repository {workspace_slug}/{repo_slug}'
        )

        return True

    async def send_response_to_pr(
        self, workspace_slug: str, repo_slug: str, pr_id: int, message: str
    ) -> bool:
        forge_webhook_url = os.getenv('FORGE_APP_WEBHOOK_URL', '')

        if not forge_webhook_url:
            logger.warning('FORGE_APP_WEBHOOK_URL not configured, cannot send response')
            return False

        payload = {
            'workspace': workspace_slug,
            'repo': repo_slug,
            'prId': pr_id,
            'message': message,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    forge_webhook_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                ) as response:
                    if response.status == 200:
                        logger.info(
                            f'Successfully sent response to PR #{pr_id} in {workspace_slug}/{repo_slug}'
                        )
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(
                            f'Failed to send response to Forge app. Status: {response.status}, '
                            f'Error: {error_text}'
                        )
                        return False

        except Exception as e:
            logger.error(f'Error sending response to Forge app: {str(e)}')
            return False
