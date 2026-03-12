from typing import Any

import jwt
from integrations.manager import Manager
from integrations.models import Message, SourceType
from integrations.slack.slack_errors import SlackError, SlackErrorCode
from integrations.slack.slack_types import (
    SlackMessageView,
    SlackViewInterface,
    StartingConvoException,
)
from integrations.slack.slack_view import (
    SlackFactory,
    SlackNewConversationFromRepoFormView,
    SlackNewConversationView,
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

    async def _store_user_msg_for_form(
        self, message_ts: str, thread_ts: str | None, user_msg: str
    ) -> None:
        """Store user message in Redis for later retrieval when form is submitted.

        This is needed because when a user selects a repo from the external_select
        dropdown, Slack sends a separate interaction payload that doesn't include
        the original user message.

        Args:
            message_ts: The message timestamp (unique identifier)
            thread_ts: The thread timestamp (if in a thread)
            user_msg: The original user message to store

        Raises:
            SlackError: If storage fails (REDIS_STORE_FAILED)
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
            raise SlackError(
                SlackErrorCode.REDIS_STORE_FAILED,
                log_context={'message_ts': message_ts, 'thread_ts': thread_ts},
            )

    async def _retrieve_user_msg_for_form(
        self, message_ts: str, thread_ts: str | None
    ) -> str:
        """Retrieve stored user message from Redis.

        Args:
            message_ts: The message timestamp
            thread_ts: The thread timestamp (if in a thread)

        Returns:
            The stored user message

        Raises:
            SlackError: If retrieval fails (REDIS_RETRIEVE_FAILED) or message
                        not found (SESSION_EXPIRED)
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
                return user_msg
            else:
                logger.warning(
                    'slack_user_msg_not_found',
                    extra={
                        'message_ts': message_ts,
                        'thread_ts': thread_ts,
                        'key': key,
                    },
                )
                raise SlackError(
                    SlackErrorCode.SESSION_EXPIRED,
                    log_context={'message_ts': message_ts, 'thread_ts': thread_ts},
                )
        except SlackError:
            raise
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
            raise SlackError(
                SlackErrorCode.REDIS_RETRIEVE_FAILED,
                log_context={'message_ts': message_ts, 'thread_ts': thread_ts},
            )

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

    def _build_repo_options(self, repos: list[Repository]) -> list[dict[str, Any]]:
        """Build Slack options list from repositories.

        Always includes a "No Repository" option at the top, followed by up to 99
        repositories (Slack has a 100 option limit for external_select).

        Args:
            repos: List of Repository objects

        Returns:
            List of Slack option objects
        """
        options: list[dict[str, Any]] = [
            {
                'text': {'type': 'plain_text', 'text': 'No Repository'},
                'value': '-',
            }
        ]
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

    async def search_repos_for_slack(
        self, user_auth: UserAuth, query: str, per_page: int = 20
    ) -> list[dict[str, Any]]:
        """Public API for repository search with formatted Slack options.

        This method searches for repositories and formats the results as Slack
        external_select options.

        Args:
            user_auth: The user's authentication context
            query: Search query to filter repositories (empty string returns all)
            per_page: Maximum number of results to return (default: 20)

        Returns:
            List of Slack option objects ready for external_select response
        """
        repos = await self._search_repositories(
            user_auth, query=query, per_page=per_page
        )
        return self._build_repo_options(repos)

    async def receive_message(self, message: Message):
        """Process an incoming Slack message.

        This is the single entry point for all Slack message processing.
        All SlackErrors raised during processing are caught and handled here,
        sending appropriate error messages to the user.
        """
        self._confirm_incoming_source_type(message)

        try:
            slack_view = await self._process_message(message)
            if slack_view and await self.is_job_requested(message, slack_view):
                await self.start_job(slack_view)

        except SlackError as e:
            await self.handle_slack_error(message.message, e)

        except Exception as e:
            logger.exception(
                'slack_unexpected_error',
                extra={'error': str(e), **message.message},
            )
            await self.handle_slack_error(
                message.message,
                SlackError(SlackErrorCode.UNEXPECTED_ERROR),
            )

    async def receive_form_interaction(self, slack_payload: dict):
        """Process a Slack form interaction (repository selection).

        This handles the block_actions payload when a user selects a repository
        from the dropdown form. It retrieves the original user message from Redis
        and delegates to receive_message for processing.

        Args:
            slack_payload: The raw Slack interaction payload
        """
        # Extract fields from the Slack interaction payload
        selected_repository = slack_payload['actions'][0]['selected_option']['value']
        if selected_repository == '-':
            selected_repository = None

        slack_user_id = slack_payload['user']['id']
        channel_id = slack_payload['container']['channel_id']
        team_id = slack_payload['team']['id']

        # Get original message_ts and thread_ts from action_id
        attribs = slack_payload['actions'][0]['action_id'].split('repository_select:')[
            -1
        ]
        message_ts, thread_ts = attribs.split(':')
        thread_ts = None if thread_ts == 'None' else thread_ts

        # Build partial payload for error handling during Redis retrieval
        payload = {
            'team_id': team_id,
            'channel_id': channel_id,
            'slack_user_id': slack_user_id,
            'message_ts': message_ts,
            'thread_ts': thread_ts,
        }

        # Retrieve the original user message from Redis
        try:
            user_msg = await self._retrieve_user_msg_for_form(message_ts, thread_ts)
        except SlackError as e:
            await self.handle_slack_error(payload, e)
            return
        except Exception as e:
            logger.exception(
                'slack_unexpected_error',
                extra={'error': str(e), **payload},
            )
            await self.handle_slack_error(
                payload, SlackError(SlackErrorCode.UNEXPECTED_ERROR)
            )
            return

        # Complete the payload and delegate to receive_message
        payload['selected_repo'] = selected_repository
        payload['user_msg'] = user_msg

        message = Message(source=SourceType.SLACK, message=payload)
        await self.receive_message(message)

    async def _process_message(self, message: Message) -> SlackViewInterface | None:
        """Process message and return view if authenticated, or raise SlackError.

        Returns:
            SlackViewInterface if user is authenticated and ready to proceed,
            None if processing should stop (but no error).

        Raises:
            SlackError: If user is not authenticated or other recoverable error.
        """
        slack_user, saas_user_auth = await self.authenticate_user(
            slack_user_id=message.message['slack_user_id']
        )

        slack_view = await SlackFactory.create_slack_view_from_payload(
            message, slack_user, saas_user_auth
        )

        # Check if this is an unauthenticated user (SlackMessageView but not SlackViewInterface)
        if not isinstance(slack_view, SlackViewInterface):
            login_link = self._generate_login_link_with_state(message)
            raise SlackError(
                SlackErrorCode.USER_NOT_AUTHENTICATED,
                message_kwargs={'login_link': login_link},
                log_context=slack_view.to_log_context(),
            )

        return slack_view

    def _generate_login_link_with_state(self, message: Message) -> str:
        """Generate OAuth login link with message state encoded."""
        jwt_secret = config.jwt_secret
        if not jwt_secret:
            raise ValueError('Must configure jwt_secret')
        state = jwt.encode(
            message.message, jwt_secret.get_secret_value(), algorithm='HS256'
        )
        return authorize_url_generator.generate(state)

    async def handle_slack_error(self, payload: dict, error: SlackError) -> None:
        """Handle a SlackError by logging and sending user message.

        This is the centralized error handler for all SlackErrors, used by both
        the manager and routes.

        Args:
            payload: The Slack payload dict containing channel/user info
            error: The SlackError to handle
        """
        # Create a minimal view for sending the error message
        view = await SlackMessageView.from_payload(
            payload, self._get_slack_team_store()
        )

        if not view:
            logger.error(
                'slack_error_no_view',
                extra={
                    'error_code': error.code.value,
                    **error.log_context,
                },
            )
            return

        # Log the error
        log_level = (
            'exception' if error.code == SlackErrorCode.UNEXPECTED_ERROR else 'warning'
        )
        log_data = {
            'error_code': error.code.value,
            **view.to_log_context(),
            **error.log_context,
        }
        getattr(logger, log_level)(
            f'slack_error_{error.code.name.lower()}', extra=log_data
        )

        # Send user-facing message
        await self.send_message(error.get_user_message(), view, ephemeral=True)

    def _get_slack_team_store(self):
        """Get the SlackTeamStore instance (lazy import to avoid circular deps)."""
        from storage.slack_team_store import SlackTeamStore

        return SlackTeamStore.get_instance()

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
    ) -> None:
        """Display the repository selection form to the user.

        Raises:
            SlackError: If storing the user message fails (REDIS_STORE_FAILED)
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

        # Store the user message for later retrieval - raises SlackError on failure
        await self._store_user_msg_for_form(
            slack_view.message_ts, slack_view.thread_ts, slack_view.user_msg
        )

        repo_selection_msg = {
            'text': 'Choose a Repository:',
            'blocks': self._generate_repo_selection_form(
                slack_view.message_ts, slack_view.thread_ts
            ),
        }
        await self.send_message(repo_selection_msg, slack_view, ephemeral=True)

    async def is_job_requested(
        self, message: Message, slack_view: SlackViewInterface
    ) -> bool:
        """Determine if a job should be started based on the current context.

        This method checks:
            1. If the view type allows immediate job start
            2. If a repo can be inferred and verified from the message
            3. Otherwise shows the repo selection form

        Args:
            slack_view: Must be a SlackViewType (authenticated view that can start jobs)

        Returns:
            True if job should start, False if waiting for user input
        """
        # Check if view type allows immediate start
        if isinstance(slack_view, SlackUpdateExistingConversationView):
            return True
        if isinstance(slack_view, SlackNewConversationFromRepoFormView):
            return True

        # For new conversations, try to infer/verify repo or show selection form
        if isinstance(slack_view, SlackNewConversationView):
            if await self._try_verify_inferred_repo(slack_view):
                return True
            await self._show_repo_selection_form(slack_view)

        return False

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
