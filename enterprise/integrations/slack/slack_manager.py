import re

import jwt
from integrations.manager import Manager
from integrations.models import Message, SourceType
from integrations.slack.slack_types import SlackViewInterface, StartingConvoException
from integrations.slack.slack_view import (
    SlackFactory,
    SlackNewConversationFromRepoFormView,
    SlackNewConversationView,
    SlackUnkownUserView,
    SlackUpdateExistingConversationView,
)
from integrations.utils import (
    HOST_URL,
    OPENHANDS_RESOLVER_TEMPLATES_DIR,
)
from jinja2 import Environment, FileSystemLoader
from pydantic import SecretStr
from server.auth.saas_user_auth import SaasUserAuth
from server.constants import SLACK_CLIENT_ID
from server.utils.conversation_callback_utils import register_callback_processor
from slack_sdk.oauth import AuthorizeUrlGenerator
from slack_sdk.web.async_client import AsyncWebClient
from storage.database import session_maker
from storage.slack_user import SlackUser

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.provider import ProviderHandler
from openhands.integrations.service_types import Repository
from openhands.server.shared import config, server_config
from openhands.server.types import LLMAuthenticationError, MissingSettingsError
from openhands.server.user_auth.user_auth import UserAuth

authorize_url_generator = AuthorizeUrlGenerator(
    client_id=SLACK_CLIENT_ID,
    scopes=['app_mentions:read', 'chat:write'],
    user_scopes=['search:read'],
)


class SlackManager(Manager):
    def __init__(self, token_manager):
        self.token_manager = token_manager
        self.login_link = (
            'User has not yet authenticated: [Click here to Login to OpenHands]({}).'
        )

        self.jinja_env = Environment(
            loader=FileSystemLoader(OPENHANDS_RESOLVER_TEMPLATES_DIR + 'slack')
        )

    def _confirm_incoming_source_type(self, message: Message):
        if message.source != SourceType.SLACK:
            raise ValueError(f'Unexpected message source {message.source}')

    async def _get_user_auth(self, keycloak_user_id: str) -> UserAuth:
        offline_token = await self.token_manager.load_offline_token(keycloak_user_id)
        if offline_token is None:
            logger.info('no_offline_token_found')

        user_auth = SaasUserAuth(
            user_id=keycloak_user_id,
            refresh_token=SecretStr(offline_token),
        )
        return user_auth

    async def authenticate_user(
        self, slack_user_id: str
    ) -> tuple[SlackUser | None, UserAuth | None]:
        # We get the user and correlate them back to a user in OpenHands - if we can
        slack_user = None
        with session_maker() as session:
            slack_user = (
                session.query(SlackUser)
                .filter(SlackUser.slack_user_id == slack_user_id)
                .first()
            )

            # slack_view.slack_to_openhands_user = slack_user # attach user auth info to view

        saas_user_auth = None
        if slack_user:
            saas_user_auth = await self._get_user_auth(slack_user.keycloak_user_id)
            # slack_view.saas_user_auth = await self._get_user_auth(slack_view.slack_to_openhands_user.keycloak_user_id)

        return slack_user, saas_user_auth

    def _infer_repo_from_message(self, user_msg: str) -> str | None:
        # Regular expression to match patterns like "OpenHands/OpenHands" or "deploy repo"
        pattern = r'([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)|([a-zA-Z0-9_-]+)(?=\s+repo)'
        match = re.search(pattern, user_msg)

        if match:
            repo = match.group(1) if match.group(1) else match.group(2)
            return repo

        return None

    async def _get_repositories(self, user_auth: UserAuth) -> list[Repository]:
        provider_tokens = await user_auth.get_provider_tokens()
        if provider_tokens is None:
            return []
        access_token = await user_auth.get_access_token()
        user_id = await user_auth.get_user_id()
        client = ProviderHandler(
            provider_tokens=provider_tokens,
            external_auth_token=access_token,
            external_auth_id=user_id,
        )
        repos: list[Repository] = await client.get_repositories(
            'pushed', server_config.app_mode, None, None, None, None
        )
        return repos

    def _generate_repo_selection_form(
        self, repo_list: list[Repository], message_ts: str, thread_ts: str | None
    ):
        options = [
            {
                'text': {'type': 'plain_text', 'text': 'No Repository'},
                'value': '-',
            }
        ]
        options.extend(
            {
                'text': {
                    'type': 'plain_text',
                    'text': repo.full_name,
                },
                'value': repo.full_name,
            }
            for repo in repo_list
        )

        return [
            {
                'type': 'header',
                'text': {
                    'type': 'plain_text',
                    'text': 'Choose a repository',
                    'emoji': True,
                },
            },
            {
                'type': 'actions',
                'elements': [
                    {
                        'type': 'static_select',
                        'action_id': f'repository_select:{message_ts}:{thread_ts}',
                        'options': options,
                    }
                ],
            },
        ]

    def filter_potential_repos_by_user_msg(
        self, user_msg: str, user_repos: list[Repository]
    ) -> tuple[bool, list[Repository]]:
        inferred_repo = self._infer_repo_from_message(user_msg)
        if not inferred_repo:
            return False, user_repos[0:99]

        final_repos = []
        for repo in user_repos:
            if inferred_repo.lower() in repo.full_name.lower():
                final_repos.append(repo)

        # no repos matched, return original list
        if len(final_repos) == 0:
            return False, user_repos[0:99]

        # Found exact match
        elif len(final_repos) == 1:
            return True, final_repos

        # Found partial matches
        return False, final_repos[0:99]

    async def receive_message(self, message: Message):
        self._confirm_incoming_source_type(message)

        slack_user, saas_user_auth = await self.authenticate_user(
            slack_user_id=message.message['slack_user_id']
        )

        try:
            slack_view = SlackFactory.create_slack_view_from_payload(
                message, slack_user, saas_user_auth
            )
        except Exception as e:
            logger.error(
                f'[Slack]: Failed to create slack view: {e}',
                exc_info=True,
                stack_info=True,
            )
            return

        if isinstance(slack_view, SlackUnkownUserView):
            jwt_secret = config.jwt_secret
            if not jwt_secret:
                raise ValueError('Must configure jwt_secret')
            state = jwt.encode(
                message.message, jwt_secret.get_secret_value(), algorithm='HS256'
            )
            link = authorize_url_generator.generate(state)
            msg = self.login_link.format(link)

            logger.info('slack_not_yet_authenticated')
            await self.send_message(
                self.create_outgoing_message(msg, ephemeral=True), slack_view
            )
            return

        if not await self.is_job_requested(message, slack_view):
            return

        await self.start_job(slack_view)

    async def send_message(self, message: Message, slack_view: SlackViewInterface):
        client = AsyncWebClient(token=slack_view.bot_access_token)
        if message.ephemeral and isinstance(message.message, str):
            await client.chat_postEphemeral(
                channel=slack_view.channel_id,
                markdown_text=message.message,
                user=slack_view.slack_user_id,
                thread_ts=slack_view.thread_ts,
            )
        elif message.ephemeral and isinstance(message.message, dict):
            await client.chat_postEphemeral(
                channel=slack_view.channel_id,
                user=slack_view.slack_user_id,
                thread_ts=slack_view.thread_ts,
                text=message.message['text'],
                blocks=message.message['blocks'],
            )
        else:
            await client.chat_postMessage(
                channel=slack_view.channel_id,
                markdown_text=message.message,
                thread_ts=slack_view.message_ts,
            )

    async def is_job_requested(
        self, message: Message, slack_view: SlackViewInterface
    ) -> bool:
        """
        A job is always request we only receive webhooks for events associated with the slack bot
        This method really just checks
            1. Is the user is authenticated
            2. Do we have the necessary information to start a job (either by inferring the selected repo, otherwise asking the user)
        """

        # Infer repo from user message is not needed; user selected repo from the form or is updating existing convo
        if isinstance(slack_view, SlackUpdateExistingConversationView):
            return True
        elif isinstance(slack_view, SlackNewConversationFromRepoFormView):
            return True
        elif isinstance(slack_view, SlackNewConversationView):
            user = slack_view.slack_to_openhands_user
            user_repos: list[Repository] = await self._get_repositories(
                slack_view.saas_user_auth
            )
            match, repos = self.filter_potential_repos_by_user_msg(
                slack_view.user_msg, user_repos
            )

            # User mentioned a matching repo is their message, start job without repo selection form
            if match:
                slack_view.selected_repo = repos[0].full_name
                return True

            logger.info(
                'render_repository_selector',
                extra={
                    'slack_user_id': user,
                    'keycloak_user_id': user.keycloak_user_id,
                    'message_ts': slack_view.message_ts,
                    'thread_ts': slack_view.thread_ts,
                },
            )

            repo_selection_msg = {
                'text': 'Choose a Repository:',
                'blocks': self._generate_repo_selection_form(
                    repos, slack_view.message_ts, slack_view.thread_ts
                ),
            }
            await self.send_message(
                self.create_outgoing_message(repo_selection_msg, ephemeral=True),
                slack_view,
            )

            return False

        return True

    async def start_job(self, slack_view: SlackViewInterface):
        # Importing here prevents circular import
        from server.conversation_callback_processor.slack_callback_processor import (
            SlackCallbackProcessor,
        )

        try:
            msg_info = None
            user_info: SlackUser = slack_view.slack_to_openhands_user
            try:
                logger.info(
                    f'[Slack] Starting job for user {user_info.slack_display_name} (id={user_info.slack_user_id})',
                    extra={'keyloak_user_id': user_info.keycloak_user_id},
                )
                conversation_id = await slack_view.create_or_update_conversation(
                    self.jinja_env
                )

                logger.info(
                    f'[Slack] Created conversation {conversation_id} for user {user_info.slack_display_name}'
                )

                if not isinstance(slack_view, SlackUpdateExistingConversationView):
                    # We don't re-subscribe for follow up messages from slack.
                    # Summaries are generated for every messages anyways, we only need to do
                    # this subscription once for the event which kicked off the job.
                    processor = SlackCallbackProcessor(
                        slack_user_id=slack_view.slack_user_id,
                        channel_id=slack_view.channel_id,
                        message_ts=slack_view.message_ts,
                        thread_ts=slack_view.thread_ts,
                        team_id=slack_view.team_id,
                    )

                    # Register the callback processor
                    register_callback_processor(conversation_id, processor)

                    logger.info(
                        f'[Slack] Created callback processor for conversation {conversation_id}'
                    )

                msg_info = slack_view.get_response_msg()

            except MissingSettingsError as e:
                logger.warning(
                    f'[Slack] Missing settings error for user {user_info.slack_display_name}: {str(e)}'
                )

                msg_info = f'{user_info.slack_display_name} please re-login into [OpenHands Cloud]({HOST_URL}) before starting a job.'

            except LLMAuthenticationError as e:
                logger.warning(
                    f'[Slack] LLM authentication error for user {user_info.slack_display_name}: {str(e)}'
                )

                msg_info = f'@{user_info.slack_display_name} please set a valid LLM API key in [OpenHands Cloud]({HOST_URL}) before starting a job.'

            except StartingConvoException as e:
                msg_info = str(e)

            await self.send_message(self.create_outgoing_message(msg_info), slack_view)

        except Exception:
            logger.exception('[Slack]: Error starting job')
            msg = 'Uh oh! There was an unexpected error starting the job :('
            await self.send_message(self.create_outgoing_message(msg), slack_view)
