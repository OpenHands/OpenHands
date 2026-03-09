from typing import Any

import jwt
from integrations.manager import Manager
from integrations.models import Message, SourceType
from integrations.slack.slack_types import (
    SlackMessageView,
    SlackViewInterface,
    StartingConvoException,
)
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
    get_session_expired_message,
    infer_repo_from_message,
)
from integrations.v1_utils import get_saas_user_auth
from jinja2 import Environment, FileSystemLoader
from server.constants import SLACK_CLIENT_ID
from server.utils.conversation_callback_utils import register_callback_processor
from slack_sdk.oauth import AuthorizeUrlGenerator
from slack_sdk.web.async_client import AsyncWebClient
from sqlalchemy import select
from storage.database import a_session_maker
from storage.slack_user import SlackUser

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.provider import ProviderHandler
from openhands.integrations.service_types import (
    AuthenticationError,
    ProviderTimeoutError,
    Repository,
)
from openhands.server.shared import config, server_config, sio
from openhands.server.types import (
    LLMAuthenticationError,
    MissingSettingsError,
    SessionExpiredError,
)
from openhands.server.user_auth.user_auth import UserAuth

authorize_url_generator = AuthorizeUrlGenerator(
    client_id=SLACK_CLIENT_ID,
    scopes=['app_mentions:read', 'chat:write'],
    user_scopes=['search:read'],
)

# Key prefix for storing user messages in Redis during repo selection flow
SLACK_USER_MSG_KEY_PREFIX = 'slack_user_msg'
# Expiration time for stored user messages (5 minutes)
# Arbitrary timeout based on typical user attention span; may be tuned based on feedback
SLACK_USER_MSG_EXPIRATION = 300


class SlackManager(Manager[SlackViewInterface]):
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

    async def authenticate_user(
        self, slack_user_id: str
    ) -> tuple[SlackUser | None, UserAuth | None]:
        # We get the user and correlate them back to a user in OpenHands - if we can
        slack_user = None
        async with a_session_maker() as session:
            result = await session.execute(
                select(SlackUser).where(SlackUser.slack_user_id == slack_user_id)
            )
            slack_user = result.scalar_one_or_none()

            # slack_view.slack_to_openhands_user = slack_user # attach user auth info to view

        saas_user_auth = None
        if slack_user:
            saas_user_auth = await get_saas_user_auth(
                slack_user.keycloak_user_id, self.token_manager
            )
            # slack_view.saas_user_auth = await self._get_user_auth(slack_view.slack_to_openhands_user.keycloak_user_id)

        return slack_user, saas_user_auth

    async def store_user_msg_for_form(
        self, message_ts: str, thread_ts: str | None, user_msg: str
    ) -> bool:
        """Store user message in Redis for later retrieval when form is submitted.

        This is needed because when a user selects a repo from the external_select
        dropdown, Slack sends a separate interaction payload that doesn't include
        the original user message.

        Args:
            message_ts: The message timestamp (unique identifier)
            thread_ts: The thread timestamp (if in a thread)
            user_msg: The original user message to store

        Returns:
            True if the message was stored successfully, False otherwise
        """
        key = f'{SLACK_USER_MSG_KEY_PREFIX}:{message_ts}:{thread_ts}'
        try:
            redis = sio.manager.redis
            await redis.set(key, user_msg, ex=SLACK_USER_MSG_EXPIRATION)
            logger.info(
                'slack_stored_user_msg',
                extra={
                    'message_ts': message_ts,
                    'thread_ts': thread_ts,
                    'key': key,
                },
            )
            return True
        except Exception as e:
            logger.error(
                'slack_store_user_msg_failed',
                extra={
                    'message_ts': message_ts,
                    'thread_ts': thread_ts,
                    'key': key,
                    'error': str(e),
                },
            )
            return False

    async def retrieve_user_msg_for_form(
        self, message_ts: str, thread_ts: str | None
    ) -> str | None:
        """Retrieve stored user message from Redis.

        Args:
            message_ts: The message timestamp
            thread_ts: The thread timestamp (if in a thread)

        Returns:
            The stored user message, or None if not found or on error
        """
        key = f'{SLACK_USER_MSG_KEY_PREFIX}:{message_ts}:{thread_ts}'
        try:
            redis = sio.manager.redis
            user_msg = await redis.get(key)
            if user_msg:
                # Redis returns bytes, decode to string
                if isinstance(user_msg, bytes):
                    user_msg = user_msg.decode('utf-8')
                logger.info(
                    'slack_retrieved_user_msg',
                    extra={
                        'message_ts': message_ts,
                        'thread_ts': thread_ts,
                        'key': key,
                    },
                )
            else:
                logger.warning(
                    'slack_user_msg_not_found',
                    extra={
                        'message_ts': message_ts,
                        'thread_ts': thread_ts,
                        'key': key,
                    },
                )
            return user_msg
        except Exception as e:
            logger.error(
                'slack_retrieve_user_msg_failed',
                extra={
                    'message_ts': message_ts,
                    'thread_ts': thread_ts,
                    'key': key,
                    'error': str(e),
                },
            )
            return None

    async def _search_repositories(
        self, user_auth: UserAuth, query: str = '', per_page: int = 100
    ) -> list[Repository]:
        """Search repositories for a user with optional query filtering.

        Args:
            user_auth: The user's authentication context
            query: Search query to filter repositories (empty string returns all)
            per_page: Maximum number of results to return

        Returns:
            List of matching Repository objects
        """
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
        repos: list[Repository] = await client.search_repositories(
            selected_provider=None,
            query=query,
            per_page=per_page,
            sort='pushed',
            order='desc',
            app_mode=server_config.app_mode,
        )
        return repos

    def _generate_repo_selection_form(
        self, message_ts: str, thread_ts: str | None
    ) -> list[dict[str, Any]]:
        """Generate a repo selection form using external_select for dynamic loading.

        This uses Slack's external_select element which allows:
        - Type-ahead search for repositories
        - Dynamic loading of options from an external endpoint
        - Support for users with many repositories (no 100 option limit)

        Args:
            message_ts: The message timestamp for tracking
            thread_ts: The thread timestamp if in a thread

        Returns:
            List of Slack Block Kit blocks for the selection form
        """
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
                'type': 'section',
                'text': {
                    'type': 'mrkdwn',
                    'text': 'Type to search your repositories:',
                },
            },
            {
                'type': 'actions',
                'elements': [
                    {
                        'type': 'external_select',
                        'action_id': f'repository_select:{message_ts}:{thread_ts}',
                        'placeholder': {
                            'type': 'plain_text',
                            'text': 'Search repositories...',
                        },
                        'min_query_length': 0,  # Load initial options immediately
                    }
                ],
            },
        ]

    def _build_repo_options(
        self, repos: list[Repository], include_no_repo: bool = True
    ) -> list[dict[str, Any]]:
        """Build Slack options list from repositories.

        Args:
            repos: List of Repository objects
            include_no_repo: Whether to include "No Repository" option

        Returns:
            List of Slack option objects
        """
        options: list[dict[str, Any]] = []
        if include_no_repo:
            options.append(
                {
                    'text': {'type': 'plain_text', 'text': 'No Repository'},
                    'value': '-',
                }
            )
        options.extend(
            {
                'text': {
                    'type': 'plain_text',
                    'text': repo.full_name[:75],  # Slack has 75 char limit for text
                },
                'value': repo.full_name,
            }
            for repo in repos[:99]  # Leave room for "No Repository" option
        )
        return options

    async def receive_message(self, message: Message):
        self._confirm_incoming_source_type(message)

        slack_user, saas_user_auth = await self.authenticate_user(
            slack_user_id=message.message['slack_user_id']
        )

        try:
            slack_view = await SlackFactory.create_slack_view_from_payload(
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
            await self.send_message(msg, slack_view, ephemeral=True)
            return

        if not await self.is_job_requested(message, slack_view):
            return

        await self.start_job(slack_view)

    async def send_message(
        self,
        message: str | dict[str, Any],
        slack_view: SlackMessageView,
        ephemeral: bool = False,
    ):
        """Send a message to Slack.

        Args:
            message: The message content. Can be a string (for simple text) or
                     a dict with 'text' and 'blocks' keys (for structured messages).
            slack_view: The Slack view object containing channel/thread info.
                        Can be either SlackMessageView (for unauthenticated users)
                        or SlackViewInterface (for authenticated users).
            ephemeral: If True, send as an ephemeral message visible only to the user.
        """
        client = AsyncWebClient(token=slack_view.bot_access_token)
        if ephemeral and isinstance(message, str):
            await client.chat_postEphemeral(
                channel=slack_view.channel_id,
                markdown_text=message,
                user=slack_view.slack_user_id,
                thread_ts=slack_view.thread_ts,
            )
        elif ephemeral and isinstance(message, dict):
            await client.chat_postEphemeral(
                channel=slack_view.channel_id,
                user=slack_view.slack_user_id,
                thread_ts=slack_view.thread_ts,
                text=message['text'],
                blocks=message['blocks'],
            )
        else:
            await client.chat_postMessage(
                channel=slack_view.channel_id,
                markdown_text=message,
                thread_ts=slack_view.message_ts,
            )

    @staticmethod
    async def send_ephemeral_message(
        bot_token: str,
        channel_id: str,
        user_id: str,
        message: str,
        thread_ts: str | None = None,
    ) -> bool:
        """Send an ephemeral message to a Slack user without requiring a SlackView.

        This is a standalone helper method for sending ephemeral messages when
        a full SlackView object is not available (e.g., in route handlers).

        Args:
            bot_token: The Slack bot token for the team.
            channel_id: The Slack channel ID.
            user_id: The Slack user ID to send the message to.
            message: The message text to send.
            thread_ts: Optional thread timestamp for threaded messages.

        Returns:
            True if the message was sent successfully, False otherwise.
        """
        try:
            client = AsyncWebClient(token=bot_token)
            await client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text=message,
                thread_ts=thread_ts,
            )
            return True
        except Exception as e:
            logger.error(
                'slack_send_ephemeral_message_failed',
                extra={
                    'channel_id': channel_id,
                    'user_id': user_id,
                    'error': str(e),
                },
                exc_info=True,
            )
            return False

    def generate_login_link(self, state: str = '') -> str:
        """Generate the OAuth login link for Slack authentication.

        Args:
            state: Optional state parameter for the OAuth flow.

        Returns:
            The login link message with embedded OAuth URL.
        """
        link = authorize_url_generator.generate(state)
        return self.login_link.format(link)

    def _should_start_job_immediately(
        self, slack_view: SlackViewInterface
    ) -> bool | None:
        """Check if the job should start immediately without repo selection.

        Returns:
            True if job should start (view already has repo context)
            None if further processing is needed (new conversation needs repo)
        """
        if isinstance(slack_view, SlackUpdateExistingConversationView):
            return True
        elif isinstance(slack_view, SlackNewConversationFromRepoFormView):
            return True
        return None

    async def _try_verify_inferred_repo(
        self, slack_view: SlackNewConversationView
    ) -> bool:
        """Try to infer and verify a repository from the user's message.

        Returns:
            True if a valid repo was found and verified, False otherwise
        """
        user = slack_view.slack_to_openhands_user
        inferred_repos = infer_repo_from_message(slack_view.user_msg)

        if len(inferred_repos) != 1:
            return False

        inferred_repo = inferred_repos[0]
        logger.info(
            f'[Slack] Verifying inferred repo "{inferred_repo}" '
            f'for user {user.slack_display_name} (id={slack_view.saas_user_auth.get_user_id()})'
        )

        try:
            provider_tokens = await slack_view.saas_user_auth.get_provider_tokens()
            if not provider_tokens:
                return False

            access_token = await slack_view.saas_user_auth.get_access_token()
            user_id = await slack_view.saas_user_auth.get_user_id()
            provider_handler = ProviderHandler(
                provider_tokens=provider_tokens,
                external_auth_token=access_token,
                external_auth_id=user_id,
            )
            repo = await provider_handler.verify_repo_provider(inferred_repo)
            slack_view.selected_repo = repo.full_name
            return True
        except (AuthenticationError, ProviderTimeoutError) as e:
            logger.info(
                f'[Slack] Could not verify repo "{inferred_repo}": {e}. '
                f'Showing repository selector.'
            )
            return False

    async def _show_repo_selection_form(
        self, slack_view: SlackNewConversationView
    ) -> bool:
        """Display the repository selection form to the user.

        Returns:
            False (job should not start yet - waiting for user selection)
        """
        user = slack_view.slack_to_openhands_user
        logger.info(
            'render_repository_selector',
            extra={
                'slack_user_id': user.slack_user_id,
                'keycloak_user_id': user.keycloak_user_id,
                'message_ts': slack_view.message_ts,
                'thread_ts': slack_view.thread_ts,
            },
        )

        store_success = await self.store_user_msg_for_form(
            slack_view.message_ts, slack_view.thread_ts, slack_view.user_msg
        )
        if not store_success:
            error_msg = (
                'Sorry, we are experiencing temporary issues. Please try again later.'
            )
            await self.send_message(error_msg, slack_view, ephemeral=True)
            return False

        repo_selection_msg = {
            'text': 'Choose a Repository:',
            'blocks': self._generate_repo_selection_form(
                slack_view.message_ts, slack_view.thread_ts
            ),
        }
        await self.send_message(repo_selection_msg, slack_view, ephemeral=True)
        return False

    async def is_job_requested(
        self, message: Message, slack_view: SlackViewInterface
    ) -> bool:
        """Determine if a job should be started based on the current context.

        This method checks:
            1. If the view type allows immediate job start
            2. If a repo can be inferred and verified from the message
            3. Otherwise shows the repo selection form

        Returns:
            True if job should start, False if waiting for user input
        """
        # Check if view type allows immediate start
        immediate_start = self._should_start_job_immediately(slack_view)
        if immediate_start is not None:
            return immediate_start

        # For new conversations, try to infer/verify repo or show selection form
        if isinstance(slack_view, SlackNewConversationView):
            if await self._try_verify_inferred_repo(slack_view):
                return True
            return await self._show_repo_selection_form(slack_view)

        return True

    async def start_job(self, slack_view: SlackViewInterface) -> None:
        # Importing here prevents circular import
        from server.conversation_callback_processor.slack_callback_processor import (
            SlackCallbackProcessor,
        )

        try:
            msg_info = None
            user_info = slack_view.slack_to_openhands_user
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

                # Only add SlackCallbackProcessor for new conversations (not updates) and non-v1 conversations
                if (
                    not isinstance(slack_view, SlackUpdateExistingConversationView)
                    and not slack_view.v1_enabled
                ):
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
                elif isinstance(slack_view, SlackUpdateExistingConversationView):
                    logger.info(
                        f'[Slack] Skipping callback processor for existing conversation update {conversation_id}'
                    )
                elif slack_view.v1_enabled:
                    logger.info(
                        f'[Slack] Skipping callback processor for v1 conversation {conversation_id}'
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

            except SessionExpiredError as e:
                logger.warning(
                    f'[Slack] Session expired for user {user_info.slack_display_name}: {str(e)}'
                )

                msg_info = get_session_expired_message(user_info.slack_display_name)

            except StartingConvoException as e:
                msg_info = str(e)

            await self.send_message(msg_info, slack_view)

        except Exception:
            logger.exception('[Slack]: Error starting job')
            await self.send_message(
                'Uh oh! There was an unexpected error starting the job :(', slack_view
            )
