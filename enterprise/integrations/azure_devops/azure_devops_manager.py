from __future__ import annotations

from types import MappingProxyType
from typing import cast

from integrations.azure_devops.azure_devops_service import SaaSAzureDevOpsService
from integrations.azure_devops.azure_devops_view import (
    AzureDevOpsFactory,
    AzureDevOpsPRComment,
    AzureDevOpsViewType,
    actor_email,
    mark_openhands_comment,
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

from openhands.app_server.integrations.azure_devops.azure_devops_service import (
    AzureDevOpsServiceImpl,
)
from openhands.app_server.integrations.provider import ProviderToken, ProviderType
from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.types import (
    LLMAuthenticationError,
    MissingSettingsError,
    SessionExpiredError,
)
from openhands.app_server.utils.logger import openhands_logger as logger


class AzureDevOpsManager(Manager[AzureDevOpsViewType]):
    def __init__(self, token_manager: TokenManager, data_collector: None = None):
        self.token_manager = token_manager
        self.jinja_env = Environment(
            loader=FileSystemLoader(OPENHANDS_RESOLVER_TEMPLATES_DIR + 'azure_devops')
        )

    def _confirm_incoming_source_type(self, message: Message) -> None:
        if message.source != SourceType.AZURE_DEVOPS:
            raise ValueError(f'Unexpected message source {message.source}')

    async def _resolve_mentioner_keycloak_id(self, message: Message) -> str | None:
        actor = AzureDevOpsFactory.extract_actor(message)
        actor_id = str(actor.get('id') or '')
        if actor_id:
            try:
                keycloak_id = await self.token_manager.get_user_id_from_idp_user_id(
                    actor_id, ProviderType.AZURE_DEVOPS
                )
                if keycloak_id:
                    return keycloak_id
            except Exception as e:
                logger.info(
                    f'[Azure DevOps] Keycloak id lookup failed for actor {actor_id}: {e}'
                )

        email = actor_email(actor)
        if email:
            try:
                return await self.token_manager.get_user_id_from_user_email(email)
            except Exception as e:
                logger.info(
                    f'[Azure DevOps] Keycloak email lookup failed for actor '
                    f'{email}: {e}'
                )

        return None

    def is_job_requested(self, message: Message) -> bool:
        self._confirm_incoming_source_type(message)
        return AzureDevOpsFactory.is_pr_comment(
            message
        ) or AzureDevOpsFactory.is_work_item_comment(message)

    async def receive_message(self, message: Message) -> None:
        self._confirm_incoming_source_type(message)
        if not self.is_job_requested(message):
            return

        keycloak_user_id = await self._resolve_mentioner_keycloak_id(message)
        actor = AzureDevOpsFactory.extract_actor(message)
        if not keycloak_user_id:
            logger.info(
                f'[Azure DevOps] Mentioner {actor.get("displayName") or actor.get("uniqueName") or "unknown"} '
                'has no OpenHands account; ignoring event.'
            )
            return

        azure_view = await AzureDevOpsFactory.create_azure_devops_view_from_payload(
            message,
            keycloak_user_id=keycloak_user_id,
        )
        if azure_view is None:
            logger.info('[Azure DevOps] No repository resolved for event; ignoring.')
            return

        # Gate on write access, mirroring the GitHub/GitLab/Bitbucket resolvers.
        # The lazy proxy is typed as the base; the enterprise impl adds the method.
        azure_service = cast(
            SaaSAzureDevOpsService,
            AzureDevOpsServiceImpl(external_auth_id=keycloak_user_id),
        )
        if not await azure_service.has_contribute_access(
            azure_view.project_id, azure_view.repository_id
        ):
            logger.info(
                f'[Azure DevOps] {actor.get("displayName") or actor.get("uniqueName") or "unknown"} '
                f'lacks write access to {azure_view.full_repo_name}; ignoring event.'
            )
            return

        logger.info(
            f'[Azure DevOps] Creating job for {azure_view.user_info.username} '
            f'in {azure_view.full_repo_name}#{azure_view.issue_number}'
        )
        await self.start_job(azure_view)

    async def send_message(
        self, message: str, azure_view: ResolverViewInterface
    ) -> None:
        message = mark_openhands_comment(message)
        azure_service = AzureDevOpsServiceImpl(
            external_auth_id=azure_view.user_info.keycloak_user_id
        )

        if isinstance(azure_view, AzureDevOpsPRComment):
            if azure_view.thread_id:
                await azure_service.add_pr_comment_to_thread(
                    azure_view.full_repo_name,
                    azure_view.issue_number,
                    azure_view.thread_id,
                    message,
                )
            else:
                await azure_service.add_pr_thread(
                    azure_view.full_repo_name,
                    azure_view.issue_number,
                    message,
                )
        else:
            await azure_service.add_work_item_comment(
                azure_view.full_repo_name,
                azure_view.issue_number,
                message,
            )

    async def start_job(self, azure_view: AzureDevOpsViewType) -> None:
        try:
            try:
                user_info = azure_view.user_info
                logger.info(
                    f'[Azure DevOps] Starting job for {user_info.username} '
                    f'in {azure_view.full_repo_name}#{azure_view.issue_number}'
                )

                offline_token = await self.token_manager.load_offline_token(
                    user_info.keycloak_user_id
                )
                if not offline_token:
                    raise MissingSettingsError('Missing settings')

                user_token = await self.token_manager.get_idp_token_from_offline_token(
                    offline_token,
                    ProviderType.AZURE_DEVOPS,
                )
                if not user_token:
                    raise MissingSettingsError('Missing settings')

                secret_store = Secrets(
                    provider_tokens=MappingProxyType(
                        {
                            ProviderType.AZURE_DEVOPS: ProviderToken(
                                token=SecretStr(user_token),
                                user_id=str(user_info.user_id),
                            )
                        }
                    )
                )

                conversation_id = await azure_view.initialize_new_conversation()
                saas_user_auth = await get_saas_user_auth(
                    user_info.keycloak_user_id,
                    self.token_manager,
                )
                await azure_view.create_new_conversation(
                    self.jinja_env,
                    secret_store.provider_tokens,
                    conversation_id,
                    saas_user_auth,
                )
                conversation_link = CONVERSATION_URL.format(azure_view.conversation_id)
                msg_info = (
                    f"I'm on it! {user_info.username} can [track my progress at "
                    f'all-hands.dev]({conversation_link})'
                )
            except MissingSettingsError as e:
                logger.warning(
                    f'[Azure DevOps] Missing settings for '
                    f'{azure_view.user_info.username}: {e}'
                )
                msg_info = (
                    f'@{azure_view.user_info.username} please re-login into '
                    f'[OpenHands Cloud]({HOST_URL}) before starting a job.'
                )
            except LLMAuthenticationError as e:
                logger.warning(
                    f'[Azure DevOps] LLM authentication error for '
                    f'{azure_view.user_info.username}: {e}'
                )
                msg_info = (
                    f'@{azure_view.user_info.username} please set a valid LLM API key '
                    f'in [OpenHands Cloud]({HOST_URL}) before starting a job.'
                )
            except SessionExpiredError as e:
                logger.warning(
                    f'[Azure DevOps] Session expired for '
                    f'{azure_view.user_info.username}: {e}'
                )
                msg_info = get_session_expired_message(azure_view.user_info.username)

            await self.send_message(msg_info, azure_view)

        except Exception as e:
            logger.exception(f'[Azure DevOps] Error starting job: {e}')
            await self.send_message(
                'Uh oh! There was an unexpected error starting the job :(',
                azure_view,
            )
