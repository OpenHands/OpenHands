from __future__ import annotations

from types import MappingProxyType

from integrations.bitbucket_data_center.bitbucket_dc_view import (
    BitbucketDCFactory,
    BitbucketDCInlinePRComment,
    BitbucketDCPRComment,
    BitbucketDCViewType,
    extract_actor_slug,
)
from integrations.manager import Manager
from integrations.models import Message, SourceType
from integrations.types import ResolverViewInterface
from integrations.utils import (
    CONVERSATION_URL,
    HOST_URL,
    OPENHANDS_RESOLVER_TEMPLATES_DIR,
    get_session_expired_message,
)
from integrations.v1_utils import get_saas_user_auth
from jinja2 import Environment, FileSystemLoader
from pydantic import SecretStr
from server.auth.token_manager import TokenManager
from storage.bitbucket_dc_webhook_store import BitbucketDCWebhookStore

from openhands.app_server.integrations.provider import ProviderToken, ProviderType
from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.types import (
    LLMAuthenticationError,
    MissingSettingsError,
    SessionExpiredError,
)
from openhands.app_server.utils.logger import openhands_logger as logger


class BitbucketDCManager(Manager[BitbucketDCViewType]):
    """Resolver manager for Bitbucket Data Center webhook events."""

    def __init__(
        self,
        token_manager: TokenManager,
        data_collector: None = None,
    ) -> None:
        self.token_manager = token_manager
        self.webhook_store = BitbucketDCWebhookStore()
        # Reuse the Bitbucket Cloud Jinja templates — wording is provider-
        # agnostic and the variables (pr_number, branch_name, comments,
        # file_location, line_number) match.
        self.jinja_env = Environment(
            loader=FileSystemLoader(OPENHANDS_RESOLVER_TEMPLATES_DIR + 'bitbucket')
        )

    def _confirm_incoming_source_type(self, message: Message) -> None:
        if message.source != SourceType.BITBUCKET_DATA_CENTER:
            raise ValueError(f'Unexpected message source {message.source}')

    @staticmethod
    def _extract_repo_identity(message: Message) -> tuple[str, str]:
        payload = message.message.get('payload') or {}
        pull_request = payload.get('pullRequest') or {}
        repository = (pull_request.get('toRef') or {}).get('repository') or {}
        project = repository.get('project') or {}
        return project.get('key') or '', repository.get('slug') or ''

    async def _commenter_has_write_access(
        self, message: Message, installer_user_id: str
    ) -> bool:
        """Use the installer's Bitbucket DC token to check whether the
        commenter has ``REPO_WRITE``/``REPO_ADMIN`` permission on the PR's
        repository — mirrors the Cloud manager's installer-scoped check.
        """
        from integrations.bitbucket_data_center.bitbucket_dc_service import (
            SaaSBitbucketDCService,
        )

        project_key, repo_slug = self._extract_repo_identity(message)
        actor = (message.message.get('payload') or {}).get('actor') or {}
        actor_slug = extract_actor_slug(actor)
        if not actor_slug:
            return False

        installer_service = SaaSBitbucketDCService(external_auth_id=installer_user_id)
        try:
            return await installer_service.user_has_write_access_for(
                project_key, repo_slug, actor_slug
            )
        except Exception as e:
            logger.warning(
                f'[Bitbucket DC] permission check failed for '
                f'{project_key}/{repo_slug}: {e}'
            )
            return False

    async def receive_message(self, message: Message) -> None:
        self._confirm_incoming_source_type(message)
        if not self.is_job_requested(message):
            return

        project_key, repo_slug = self._extract_repo_identity(message)
        installer_user_id = await self.webhook_store.get_webhook_user_id(
            project_key=project_key, repo_slug=repo_slug
        )
        if not installer_user_id:
            logger.warning(
                f'[Bitbucket DC] No installer recorded for '
                f'{project_key}/{repo_slug}'
            )
            return

        if not await self._commenter_has_write_access(message, installer_user_id):
            payload = message.message.get('payload') or {}
            actor = payload.get('actor') or {}
            logger.info(
                f'[Bitbucket DC] {actor.get("displayName", "?")} lacks write '
                f'access to {project_key}/{repo_slug}; ignoring.'
            )
            return

        bitbucket_view = await BitbucketDCFactory.create_bitbucket_dc_view_from_payload(
            message, installer_user_id
        )
        logger.info(
            f'[Bitbucket DC] Creating job for {bitbucket_view.user_info.username} '
            f'in {bitbucket_view.full_repo_name}#{bitbucket_view.issue_number}'
        )
        await self.start_job(bitbucket_view)

    def is_job_requested(self, message: Message) -> bool:
        self._confirm_incoming_source_type(message)
        return BitbucketDCFactory.is_pr_comment(
            message
        ) or BitbucketDCFactory.is_pr_comment(message, inline=True)

    async def send_message(
        self, message: str, bitbucket_view: ResolverViewInterface
    ) -> None:
        from integrations.bitbucket_data_center.bitbucket_dc_service import (
            SaaSBitbucketDCService,
        )

        keycloak_user_id = bitbucket_view.user_info.keycloak_user_id
        bitbucket_service = SaaSBitbucketDCService(external_auth_id=keycloak_user_id)

        if isinstance(bitbucket_view, BitbucketDCInlinePRComment):
            await bitbucket_service.reply_to_pr_comment(
                owner=bitbucket_view.project_key,
                repo_slug=bitbucket_view.repo_slug,
                pr_id=bitbucket_view.issue_number,
                body=message,
                parent_comment_id=bitbucket_view.parent_comment_id,
                anchor={
                    'path': bitbucket_view.file_location,
                    'line': bitbucket_view.line_number,
                    'lineType': bitbucket_view.line_type,
                    'fileType': bitbucket_view.file_type,
                },
            )
        elif isinstance(bitbucket_view, BitbucketDCPRComment):
            await bitbucket_service.reply_to_pr_comment(
                owner=bitbucket_view.project_key,
                repo_slug=bitbucket_view.repo_slug,
                pr_id=bitbucket_view.issue_number,
                body=message,
                parent_comment_id=bitbucket_view.parent_comment_id,
            )
        else:
            logger.warning(
                f'[Bitbucket DC] Unsupported view type: '
                f'{type(bitbucket_view).__name__}'
            )

    async def start_job(self, bitbucket_view: BitbucketDCViewType) -> None:
        try:
            user_info = bitbucket_view.user_info
            try:
                logger.info(
                    f'[Bitbucket DC] Starting job for {user_info.username} '
                    f'in {bitbucket_view.full_repo_name}#{bitbucket_view.issue_number}'
                )

                offline_token = await self.token_manager.load_offline_token(
                    user_info.keycloak_user_id
                )
                if not offline_token:
                    logger.warning(
                        f'[Bitbucket DC] No offline token for installer '
                        f'{user_info.keycloak_user_id}'
                    )
                    raise MissingSettingsError('Missing settings')

                user_token = await self.token_manager.get_idp_token_from_offline_token(
                    offline_token, ProviderType.BITBUCKET_DATA_CENTER
                )
                if not user_token:
                    logger.warning(
                        f'[Bitbucket DC] No Bitbucket DC token for installer '
                        f'{user_info.keycloak_user_id}'
                    )
                    raise MissingSettingsError('Missing settings')

                secret_store = Secrets(
                    provider_tokens=MappingProxyType(
                        {
                            ProviderType.BITBUCKET_DATA_CENTER: ProviderToken(
                                token=SecretStr(user_token),
                                user_id=str(user_info.user_id),
                            )
                        }
                    )
                )

                conversation_id = await bitbucket_view.initialize_new_conversation()
                saas_user_auth = await get_saas_user_auth(
                    user_info.keycloak_user_id, self.token_manager
                )
                await bitbucket_view.create_new_conversation(
                    self.jinja_env,
                    secret_store.provider_tokens,
                    conversation_id,
                    saas_user_auth,
                )
                conversation_id_hex = bitbucket_view.conversation_id

                logger.info(
                    f'[Bitbucket DC] Created conversation {conversation_id_hex} '
                    f'for user {user_info.username}'
                )
                conversation_link = CONVERSATION_URL.format(conversation_id_hex)
                msg_info = (
                    f"I'm on it! {user_info.username} can [track my progress at "
                    f'all-hands.dev]({conversation_link})'
                )

            except MissingSettingsError as e:
                logger.warning(
                    f'[Bitbucket DC] Missing settings for {user_info.username}: {e}'
                )
                msg_info = (
                    f'@{user_info.username} please re-login into '
                    f'[OpenHands Cloud]({HOST_URL}) before starting a job.'
                )

            except LLMAuthenticationError as e:
                logger.warning(
                    f'[Bitbucket DC] LLM authentication error for '
                    f'{user_info.username}: {e}'
                )
                msg_info = (
                    f'@{user_info.username} please set a valid LLM API key in '
                    f'[OpenHands Cloud]({HOST_URL}) before starting a job.'
                )

            except SessionExpiredError as e:
                logger.warning(
                    f'[Bitbucket DC] Session expired for {user_info.username}: {e}'
                )
                msg_info = get_session_expired_message(user_info.username)

            await self.send_message(msg_info, bitbucket_view)

        except Exception as e:
            logger.exception(f'[Bitbucket DC] Error starting job: {e}')
            await self.send_message(
                'Uh oh! There was an unexpected error starting the job :(',
                bitbucket_view,
            )
