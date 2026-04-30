from __future__ import annotations

from types import MappingProxyType

from integrations.bitbucket.bitbucket_view import (
    BitbucketFactory,
    BitbucketInlinePRComment,
    BitbucketPRComment,
    BitbucketViewType,
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
from storage.bitbucket_webhook_store import BitbucketWebhookStore

from openhands.app_server.integrations.provider import ProviderToken, ProviderType
from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.utils.logger import openhands_logger as logger
from openhands.server.types import (
    LLMAuthenticationError,
    MissingSettingsError,
    SessionExpiredError,
)


class BitbucketManager(Manager[BitbucketViewType]):
    """Resolver manager for Bitbucket Cloud webhook events."""

    def __init__(
        self,
        token_manager: TokenManager,
        data_collector: None = None,
    ) -> None:
        self.token_manager = token_manager
        self.webhook_store = BitbucketWebhookStore()
        self.jinja_env = Environment(
            loader=FileSystemLoader(OPENHANDS_RESOLVER_TEMPLATES_DIR + 'bitbucket')
        )

    def _confirm_incoming_source_type(self, message: Message) -> None:
        if message.source != SourceType.BITBUCKET:
            raise ValueError(f'Unexpected message source {message.source}')

    async def _commenter_has_write_access(
        self, message: Message, installer_user_id: str
    ) -> bool:
        """Use the installer's Bitbucket token to check whether the commenter
        has ``write``/``admin`` permission on the PR's repository.

        Calling Bitbucket as the installer mirrors how GitHub Apps gate
        resolver triggers — the installer holds repo admin (Bitbucket
        Cloud requires it for repo webhook setup), so they can read
        per-user permission records.
        """
        from integrations.bitbucket.bitbucket_service import SaaSBitBucketService

        payload = message.message.get('payload') or {}
        repository = payload.get('repository') or {}
        full_repo_name = repository.get('full_name') or ''
        workspace, _, repo_slug = full_repo_name.partition('/')
        actor = payload.get('actor') or {}
        # ``account_id`` is the canonical Bitbucket Cloud user id; fall back
        # to the (brace-stripped) ``uuid`` when an older payload omits it.
        actor_id = (
            actor.get('account_id') or (actor.get('uuid') or '').strip('{}') or ''
        )
        if not actor_id:
            return False

        installer_service = SaaSBitBucketService(external_auth_id=installer_user_id)
        try:
            return await installer_service.user_has_write_access_for(
                workspace, repo_slug, actor_id
            )
        except Exception as e:
            logger.warning(
                f'[Bitbucket] permission check failed for {full_repo_name}: {e}'
            )
            return False

    async def receive_message(self, message: Message) -> None:
        self._confirm_incoming_source_type(message)
        if not self.is_job_requested(message):
            return

        webhook_uuid = message.message.get('installation_id') or ''
        installer_user_id = await self.webhook_store.get_webhook_user_id(
            webhook_uuid=webhook_uuid
        )
        if not installer_user_id:
            logger.warning(
                f'[Bitbucket] No installer recorded for webhook {webhook_uuid}'
            )
            return

        if not await self._commenter_has_write_access(message, installer_user_id):
            payload = message.message.get('payload') or {}
            repository = payload.get('repository') or {}
            actor = payload.get('actor') or {}
            logger.info(
                f'[Bitbucket] {actor.get("display_name", "?")} lacks write '
                f'access to {repository.get("full_name", "?")}; ignoring.'
            )
            return

        bitbucket_view = await BitbucketFactory.create_bitbucket_view_from_payload(
            message, installer_user_id
        )
        logger.info(
            f'[Bitbucket] Creating job for {bitbucket_view.user_info.username} '
            f'in {bitbucket_view.full_repo_name}#{bitbucket_view.issue_number}'
        )
        await self.start_job(bitbucket_view)

    def is_job_requested(self, message: Message) -> bool:
        self._confirm_incoming_source_type(message)
        return BitbucketFactory.is_pr_comment(
            message
        ) or BitbucketFactory.is_pr_comment(message, inline=True)

    async def send_message(
        self, message: str, bitbucket_view: ResolverViewInterface
    ) -> None:
        from integrations.bitbucket.bitbucket_service import SaaSBitBucketService

        keycloak_user_id = bitbucket_view.user_info.keycloak_user_id
        bitbucket_service = SaaSBitBucketService(external_auth_id=keycloak_user_id)

        if isinstance(bitbucket_view, BitbucketInlinePRComment):
            await bitbucket_service.reply_to_pr_comment(
                workspace=bitbucket_view.workspace,
                repo_slug=bitbucket_view.repo_slug,
                pr_id=bitbucket_view.issue_number,
                body=message,
                parent_comment_id=bitbucket_view.parent_comment_id,
                inline={
                    'path': bitbucket_view.file_location,
                    'to': bitbucket_view.line_number,
                },
            )
        elif isinstance(bitbucket_view, BitbucketPRComment):
            await bitbucket_service.reply_to_pr_comment(
                workspace=bitbucket_view.workspace,
                repo_slug=bitbucket_view.repo_slug,
                pr_id=bitbucket_view.issue_number,
                body=message,
                parent_comment_id=bitbucket_view.parent_comment_id,
            )
        else:
            logger.warning(
                f'[Bitbucket] Unsupported view type: {type(bitbucket_view).__name__}'
            )

    async def start_job(self, bitbucket_view: BitbucketViewType) -> None:
        try:
            user_info = bitbucket_view.user_info
            try:
                logger.info(
                    f'[Bitbucket] Starting job for {user_info.username} '
                    f'in {bitbucket_view.full_repo_name}#{bitbucket_view.issue_number}'
                )

                # Auth runs as the webhook installer (``user_info.keycloak_user_id``),
                # not the comment author — Bitbucket Cloud's built-in IdP can't
                # populate a per-actor Keycloak attribute, so we cannot map
                # arbitrary commenters back to a Keycloak user. The
                # commenter's Bitbucket account_id stays on
                # ``ProviderToken.user_id`` for audit/display.
                offline_token = await self.token_manager.load_offline_token(
                    user_info.keycloak_user_id
                )
                if not offline_token:
                    logger.warning(
                        f'[Bitbucket] No offline token for installer '
                        f'{user_info.keycloak_user_id}'
                    )
                    raise MissingSettingsError('Missing settings')

                user_token = await self.token_manager.get_idp_token_from_offline_token(
                    offline_token, ProviderType.BITBUCKET
                )
                if not user_token:
                    logger.warning(
                        f'[Bitbucket] No Bitbucket token for installer '
                        f'{user_info.keycloak_user_id}'
                    )
                    raise MissingSettingsError('Missing settings')

                secret_store = Secrets(
                    provider_tokens=MappingProxyType(
                        {
                            ProviderType.BITBUCKET: ProviderToken(
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
                    f'[Bitbucket] Created conversation {conversation_id_hex} '
                    f'for user {user_info.username}'
                )
                conversation_link = CONVERSATION_URL.format(conversation_id_hex)
                msg_info = (
                    f"I'm on it! {user_info.username} can [track my progress at "
                    f'all-hands.dev]({conversation_link})'
                )

            except MissingSettingsError as e:
                logger.warning(
                    f'[Bitbucket] Missing settings for {user_info.username}: {e}'
                )
                msg_info = (
                    f'@{user_info.username} please re-login into '
                    f'[OpenHands Cloud]({HOST_URL}) before starting a job.'
                )

            except LLMAuthenticationError as e:
                logger.warning(
                    f'[Bitbucket] LLM authentication error for '
                    f'{user_info.username}: {e}'
                )
                msg_info = (
                    f'@{user_info.username} please set a valid LLM API key in '
                    f'[OpenHands Cloud]({HOST_URL}) before starting a job.'
                )

            except SessionExpiredError as e:
                logger.warning(
                    f'[Bitbucket] Session expired for {user_info.username}: {e}'
                )
                msg_info = get_session_expired_message(user_info.username)

            await self.send_message(msg_info, bitbucket_view)

        except Exception as e:
            logger.exception(f'[Bitbucket] Error starting job: {e}')
            await self.send_message(
                'Uh oh! There was an unexpected error starting the job :(',
                bitbucket_view,
            )
