from dataclasses import dataclass
from uuid import UUID, uuid4

from integrations.models import Message
from integrations.resolver_context import ResolverUserContext
from integrations.slack.slack_types import SlackViewInterface, StartingConvoException
from integrations.slack.slack_v1_callback_processor import SlackV1CallbackProcessor
from integrations.utils import (
    CONVERSATION_URL,
    ENABLE_V1_SLACK_RESOLVER,
    get_final_agent_observation,
    get_user_v1_enabled_setting,
)
from jinja2 import Environment
from slack_sdk import WebClient
from storage.slack_conversation import SlackConversation
from storage.slack_conversation_store import SlackConversationStore
from storage.slack_team_store import SlackTeamStore
from storage.slack_user import SlackUser

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartRequest,
    AppConversationStartTaskStatus,
    SendMessageRequest,
)
from openhands.app_server.config import get_app_conversation_service
from openhands.app_server.sandbox.sandbox_models import SandboxStatus
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.specifiy_user_context import USER_CONTEXT_ATTR
from openhands.core.logger import openhands_logger as logger
from openhands.core.schema.agent import AgentState
from openhands.events.action import MessageAction
from openhands.events.serialization.event import event_to_dict
from openhands.integrations.provider import ProviderHandler, ProviderType
from openhands.sdk import TextContent
from openhands.server.services.conversation_service import (
    create_new_conversation,
    setup_init_conversation_settings,
)
from openhands.server.shared import ConversationStoreImpl, config, conversation_manager
from openhands.server.user_auth.user_auth import UserAuth
from openhands.storage.data_models.conversation_metadata import (
    ConversationTrigger,
)
from openhands.utils.async_utils import GENERAL_TIMEOUT, call_async_from_sync

# =================================================
# SECTION: Slack view types
# =================================================


CONTEXT_LIMIT = 21
slack_conversation_store = SlackConversationStore.get_instance()
slack_team_store = SlackTeamStore.get_instance()


async def is_v1_enabled_for_slack_resolver(user_id: str) -> bool:
    return await get_user_v1_enabled_setting(user_id) and ENABLE_V1_SLACK_RESOLVER


@dataclass
class SlackUnkownUserView(SlackViewInterface):
    bot_access_token: str
    user_msg: str | None
    slack_user_id: str
    slack_to_openhands_user: SlackUser | None
    saas_user_auth: UserAuth | None
    channel_id: str
    message_ts: str
    thread_ts: str | None
    selected_repo: str | None
    should_extract: bool
    send_summary_instruction: bool
    conversation_id: str
    team_id: str
    v1_enabled: bool

    def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        raise NotImplementedError

    async def create_or_update_conversation(self, jinja_env: Environment):
        raise NotImplementedError

    def get_response_msg(self) -> str:
        raise NotImplementedError


@dataclass
class SlackNewConversationView(SlackViewInterface):
    bot_access_token: str
    user_msg: str | None
    slack_user_id: str
    slack_to_openhands_user: SlackUser
    saas_user_auth: UserAuth
    channel_id: str
    message_ts: str
    thread_ts: str | None
    selected_repo: str | None
    should_extract: bool
    send_summary_instruction: bool
    conversation_id: str
    team_id: str
    v1_enabled: bool

    def _get_initial_prompt(self, text: str, blocks: list[dict]):
        bot_id = self._get_bot_id(blocks)
        text = text.replace(f'<@{bot_id}>', '').strip()
        return text

    def _get_bot_id(self, blocks: list[dict]) -> str:
        for block in blocks:
            type_ = block['type']
            if type_ in ('rich_text', 'rich_text_section'):
                bot_id = self._get_bot_id(block['elements'])
                if bot_id:
                    return bot_id
            if type_ == 'user':
                return block['user_id']
        return ''

    def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        """Instructions passed when conversation is first initialized"""
        user_info: SlackUser = self.slack_to_openhands_user

        messages = []
        if self.thread_ts:
            client = WebClient(token=self.bot_access_token)
            result = client.conversations_replies(
                channel=self.channel_id,
                ts=self.thread_ts,
                inclusive=True,
                latest=self.message_ts,
                limit=CONTEXT_LIMIT,  # We can be smarter about getting more context/condensing it even in the future
            )

            messages = result['messages']

        else:
            client = WebClient(token=self.bot_access_token)
            result = client.conversations_history(
                channel=self.channel_id,
                inclusive=True,
                latest=self.message_ts,
                limit=CONTEXT_LIMIT,
            )

            messages = result['messages']
            messages.reverse()

        if not messages:
            raise ValueError('Failed to fetch information from slack API')

        logger.info('got_messages_from_slack', extra={'messages': messages})

        trigger_msg = messages[-1]
        user_message = self._get_initial_prompt(
            trigger_msg['text'], trigger_msg['blocks']
        )

        conversation_instructions = ''

        if len(messages) > 1:
            messages.pop()
            text_messages = [m['text'] for m in messages if m.get('text')]
            conversation_instructions_template = jinja_env.get_template(
                'user_message_conversation_instructions.j2'
            )
            conversation_instructions = conversation_instructions_template.render(
                messages=text_messages,
                username=user_info.slack_display_name,
                conversation_url=CONVERSATION_URL,
            )

        return user_message, conversation_instructions

    def _verify_necessary_values_are_set(self):
        if not self.selected_repo:
            raise ValueError(
                'Attempting to start conversation without confirming selected repo from user'
            )

    async def save_slack_convo(self, v1_enabled: bool = False):
        if self.slack_to_openhands_user:
            user_info: SlackUser = self.slack_to_openhands_user

            logger.info(
                'Create slack conversation',
                extra={
                    'channel_id': self.channel_id,
                    'conversation_id': self.conversation_id,
                    'keycloak_user_id': user_info.keycloak_user_id,
                    'org_id': user_info.org_id,
                    'parent_id': self.thread_ts or self.message_ts,
                    'v1_enabled': v1_enabled,
                },
            )
            slack_conversation = SlackConversation(
                conversation_id=self.conversation_id,
                channel_id=self.channel_id,
                keycloak_user_id=user_info.keycloak_user_id,
                org_id=user_info.org_id,
                parent_id=self.thread_ts
                or self.message_ts,  # conversations can start in a thread reply as well; we should always references the parent's (root level msg's) message ID
                v1_enabled=v1_enabled,
            )
            await slack_conversation_store.create_slack_conversation(slack_conversation)

    def _create_slack_v1_callback_processor(self) -> SlackV1CallbackProcessor:
        """Create a SlackV1CallbackProcessor for V1 conversation handling."""
        return SlackV1CallbackProcessor(
            slack_view_data={
                'channel_id': self.channel_id,
                'message_ts': self.message_ts,
                'thread_ts': self.thread_ts,
                'team_id': self.team_id,
                'slack_user_id': self.slack_user_id,
            }
        )

    async def create_or_update_conversation(self, jinja: Environment) -> str:
        """Only creates a new conversation"""
        self._verify_necessary_values_are_set()

        provider_tokens = await self.saas_user_auth.get_provider_tokens()
        user_secrets = await self.saas_user_auth.get_secrets()

        # Check if V1 conversations are enabled for this user
        self.v1_enabled = await is_v1_enabled_for_slack_resolver(
            self.slack_to_openhands_user.keycloak_user_id
        )

        if self.v1_enabled:
            # Use V1 app conversation service
            await self._create_v1_conversation(jinja)
            return self.conversation_id
        else:
            # Use existing V0 conversation service
            await self._create_v0_conversation(jinja, provider_tokens, user_secrets)
            return self.conversation_id

    async def _create_v0_conversation(
        self, jinja: Environment, provider_tokens, user_secrets
    ) -> None:
        """Create conversation using the legacy V0 system."""
        user_instructions, conversation_instructions = self._get_instructions(jinja)

        # Determine git provider from repository
        git_provider = None
        if self.selected_repo and provider_tokens:
            provider_handler = ProviderHandler(provider_tokens)
            repository = await provider_handler.verify_repo_provider(self.selected_repo)
            git_provider = repository.git_provider

        agent_loop_info = await create_new_conversation(
            user_id=self.slack_to_openhands_user.keycloak_user_id,
            git_provider_tokens=provider_tokens,
            selected_repository=self.selected_repo,
            selected_branch=None,
            initial_user_msg=user_instructions,
            conversation_instructions=(
                conversation_instructions if conversation_instructions else None
            ),
            image_urls=None,
            replay_json=None,
            conversation_trigger=ConversationTrigger.SLACK,
            custom_secrets=user_secrets.custom_secrets if user_secrets else None,
            git_provider=git_provider,
        )

        self.conversation_id = agent_loop_info.conversation_id
        logger.info(f'[Slack]: Created V0 conversation: {self.conversation_id}')
        await self.save_slack_convo(v1_enabled=False)

    async def _create_v1_conversation(self, jinja: Environment) -> None:
        """Create conversation using the new V1 app conversation system."""
        user_instructions, conversation_instructions = self._get_instructions(jinja)

        # Create the initial message request
        initial_message = SendMessageRequest(
            role='user', content=[TextContent(text=user_instructions)]
        )

        # Create the Slack V1 callback processor
        slack_callback_processor = self._create_slack_v1_callback_processor()

        # Determine git provider from repository
        git_provider = None
        provider_tokens = await self.saas_user_auth.get_provider_tokens()
        if self.selected_repo and provider_tokens:
            provider_handler = ProviderHandler(provider_tokens)
            repository = await provider_handler.verify_repo_provider(self.selected_repo)
            git_provider = ProviderType(repository.git_provider.value)

        # Get the app conversation service and start the conversation
        injector_state = InjectorState()

        # Create the V1 conversation start request with the callback processor
        self.conversation_id = uuid4().hex
        start_request = AppConversationStartRequest(
            conversation_id=UUID(self.conversation_id),
            system_message_suffix=conversation_instructions,
            initial_message=initial_message,
            selected_repository=self.selected_repo,
            git_provider=git_provider,
            title=f'Slack conversation from {self.slack_to_openhands_user.slack_display_name}',
            trigger=ConversationTrigger.SLACK,
            processors=[
                slack_callback_processor
            ],  # Pass the callback processor directly
        )

        # Set up the Slack user context for the V1 system
        slack_user_context = ResolverUserContext(saas_user_auth=self.saas_user_auth)
        setattr(injector_state, USER_CONTEXT_ATTR, slack_user_context)

        async with get_app_conversation_service(
            injector_state
        ) as app_conversation_service:
            async for task in app_conversation_service.start_app_conversation(
                start_request
            ):
                if task.status == AppConversationStartTaskStatus.ERROR:
                    logger.error(f'Failed to start V1 conversation: {task.detail}')
                    raise RuntimeError(
                        f'Failed to start V1 conversation: {task.detail}'
                    )

        logger.info(f'[Slack V1]: Created new conversation: {self.conversation_id}')
        await self.save_slack_convo(v1_enabled=True)

    def get_response_msg(self) -> str:
        user_info: SlackUser = self.slack_to_openhands_user
        conversation_link = CONVERSATION_URL.format(self.conversation_id)
        return f"I'm on it! {user_info.slack_display_name} can [track my progress here]({conversation_link})."


@dataclass
class SlackNewConversationFromRepoFormView(SlackNewConversationView):
    def _verify_necessary_values_are_set(self):
        # Exclude selected repo check from parent
        # User can start conversations without a repo when specified via the repo selection form
        return


@dataclass
class SlackUpdateExistingConversationView(SlackNewConversationView):
    slack_conversation: SlackConversation

    def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        client = WebClient(token=self.bot_access_token)
        result = client.conversations_replies(
            channel=self.channel_id,
            ts=self.message_ts,
            inclusive=True,
            latest=self.message_ts,
            limit=1,  # Get exact user message, in future we can be smarter with collecting additional context
        )

        user_message = result['messages'][0]
        user_message = self._get_initial_prompt(
            user_message['text'], user_message['blocks']
        )

        return user_message, ''

    async def send_message_to_v0_conversation(self, jinja: Environment):
        user_info: SlackUser = self.slack_to_openhands_user
        user_id = user_info.keycloak_user_id
        saas_user_auth: UserAuth = self.saas_user_auth
        provider_tokens = await saas_user_auth.get_provider_tokens()

        try:
            conversation_store = await ConversationStoreImpl.get_instance(
                config, user_id
            )
            await conversation_store.get_metadata(self.conversation_id)
        except FileNotFoundError:
            raise StartingConvoException('Conversation no longer exists.')

        # Should we raise here if there are no provider tokens?
        providers_set = list(provider_tokens.keys()) if provider_tokens else []

        conversation_init_data = await setup_init_conversation_settings(
            user_id, self.conversation_id, providers_set
        )

        # Either join ongoing conversation, or restart the conversation
        agent_loop_info = await conversation_manager.maybe_start_agent_loop(
            self.conversation_id, conversation_init_data, user_id
        )

        final_agent_observation = get_final_agent_observation(
            agent_loop_info.event_store
        )
        agent_state = (
            None
            if len(final_agent_observation) == 0
            else final_agent_observation[0].agent_state
        )

        if not agent_state or agent_state == AgentState.LOADING:
            raise StartingConvoException('Conversation is still starting')

        instructions, _ = self._get_instructions(jinja)
        user_msg = MessageAction(content=instructions)
        await conversation_manager.send_event_to_conversation(
            self.conversation_id, event_to_dict(user_msg)
        )

    async def send_message_to_v1_conversation(self, jinja: Environment):
        """Send a message to a v1 conversation using the agent server API."""
        # Import services within the method to avoid circular imports
        from openhands.agent_server.models import SendMessageRequest
        from openhands.app_server.config import (
            get_app_conversation_info_service,
            get_httpx_client,
            get_sandbox_service,
        )
        from openhands.app_server.event_callback.util import (
            ensure_conversation_found,
            get_agent_server_url_from_sandbox,
        )
        from openhands.app_server.services.injector import InjectorState
        from openhands.app_server.user.specifiy_user_context import (
            ADMIN,
            USER_CONTEXT_ATTR,
        )

        # Create injector state for dependency injection
        state = InjectorState()
        setattr(state, USER_CONTEXT_ATTR, ADMIN)

        async with (
            get_app_conversation_info_service(state) as app_conversation_info_service,
            get_sandbox_service(state) as sandbox_service,
            get_httpx_client(state) as httpx_client,
        ):
            # 1. Conversation lookup
            app_conversation_info = ensure_conversation_found(
                await app_conversation_info_service.get_app_conversation_info(
                    UUID(self.conversation_id)
                ),
                UUID(self.conversation_id),
            )

            # 2. Sandbox lookup + validation
            sandbox = await sandbox_service.get_sandbox(
                app_conversation_info.sandbox_id
            )

            if sandbox and sandbox.status == SandboxStatus.PAUSED:
                # Resume paused sandbox and wait for it to be running
                logger.info('[Slack V1]: Attempting to resume paused sandbox')
                await sandbox_service.resume_sandbox(app_conversation_info.sandbox_id)

            # Wait for sandbox to be running (handles both fresh start and resume)
            running_sandbox = await sandbox_service.wait_for_sandbox_running(
                app_conversation_info.sandbox_id,
                timeout=120,
                poll_interval=2,
                httpx_client=httpx_client,
            )

            assert (
                running_sandbox.session_api_key is not None
            ), f'No session API key for sandbox: {running_sandbox.id}'

            # 3. Get the agent server URL
            agent_server_url = get_agent_server_url_from_sandbox(running_sandbox)

            # 4. Prepare the message content
            user_msg, _ = self._get_instructions(jinja)

            # 5. Create the message request
            send_message_request = SendMessageRequest(
                role='user', content=[TextContent(text=user_msg)], run=True
            )

            # 6. Send the message to the agent server
            url = f'{agent_server_url.rstrip("/")}/api/conversations/{UUID(self.conversation_id)}/events'

            headers = {'X-Session-API-Key': running_sandbox.session_api_key}
            payload = send_message_request.model_dump()

            try:
                response = await httpx_client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=30.0,
                )
                response.raise_for_status()

            except Exception as e:
                logger.error(
                    '[Slack V1] Failed to send message to conversation %s: %s',
                    self.conversation_id,
                    str(e),
                    exc_info=True,
                )
                raise Exception(f'Failed to send message to v1 conversation: {str(e)}')

    async def create_or_update_conversation(self, jinja: Environment) -> str:
        """Send new user message to converation"""
        user_info: SlackUser = self.slack_to_openhands_user

        user_id = user_info.keycloak_user_id

        # Org management in the future will get rid of this
        # For now, only user that created the conversation can send follow up messages to it
        if user_id != self.slack_conversation.keycloak_user_id:
            raise StartingConvoException(
                f'{user_info.slack_display_name} is not authorized to send messages to this conversation.'
            )

        if self.slack_conversation.v1_enabled:
            await self.send_message_to_v1_conversation(jinja)
        else:
            await self.send_message_to_v0_conversation(jinja)

        return self.conversation_id

    def get_response_msg(self):
        user_info: SlackUser = self.slack_to_openhands_user
        conversation_link = CONVERSATION_URL.format(self.conversation_id)
        return f"I'm on it! {user_info.slack_display_name} can [continue tracking my progress here]({conversation_link})."


class SlackFactory:
    @staticmethod
    def did_user_select_repo_from_form(message: Message):
        payload = message.message
        return 'selected_repo' in payload

    @staticmethod
    async def determine_if_updating_existing_conversation(
        message: Message,
    ) -> SlackConversation | None:
        payload = message.message
        channel_id = payload.get('channel_id')
        thread_ts = payload.get('thread_ts')

        # Follow up conversations must be contained in-thread
        if not thread_ts:
            return None

        # thread_ts in slack payloads in the parent's (root level msg's) message ID
        return await slack_conversation_store.get_slack_conversation(
            channel_id, thread_ts
        )

    def create_slack_view_from_payload(
        message: Message, slack_user: SlackUser | None, saas_user_auth: UserAuth | None
    ):
        payload = message.message
        slack_user_id = payload['slack_user_id']
        channel_id = payload.get('channel_id')
        message_ts = payload.get('message_ts')
        thread_ts = payload.get('thread_ts')
        team_id = payload['team_id']
        user_msg = payload.get('user_msg')

        bot_access_token = slack_team_store.get_team_bot_token(team_id)
        if not bot_access_token:
            logger.error(
                'Did not find slack team',
                extra={
                    'slack_user_id': slack_user_id,
                    'channel_id': channel_id,
                },
            )
            raise Exception('Did not find slack team')

        # Determine if this is a known slack user by openhands
        if not slack_user or not saas_user_auth or not channel_id:
            return SlackUnkownUserView(
                bot_access_token=bot_access_token,
                user_msg=user_msg,
                slack_user_id=slack_user_id,
                slack_to_openhands_user=slack_user,
                saas_user_auth=saas_user_auth,
                channel_id=channel_id,
                message_ts=message_ts,
                thread_ts=thread_ts,
                selected_repo=None,
                should_extract=False,
                send_summary_instruction=False,
                conversation_id='',
                team_id=team_id,
                v1_enabled=False,
            )

        conversation: SlackConversation | None = call_async_from_sync(
            SlackFactory.determine_if_updating_existing_conversation,
            GENERAL_TIMEOUT,
            message,
        )
        if conversation:
            logger.info(
                'Found existing slack conversation',
                extra={
                    'conversation_id': conversation.conversation_id,
                    'parent_id': conversation.parent_id,
                },
            )
            return SlackUpdateExistingConversationView(
                bot_access_token=bot_access_token,
                user_msg=user_msg,
                slack_user_id=slack_user_id,
                slack_to_openhands_user=slack_user,
                saas_user_auth=saas_user_auth,
                channel_id=channel_id,
                message_ts=message_ts,
                thread_ts=thread_ts,
                selected_repo=None,
                should_extract=True,
                send_summary_instruction=True,
                conversation_id=conversation.conversation_id,
                slack_conversation=conversation,
                team_id=team_id,
                v1_enabled=False,
            )

        elif SlackFactory.did_user_select_repo_from_form(message):
            return SlackNewConversationFromRepoFormView(
                bot_access_token=bot_access_token,
                user_msg=user_msg,
                slack_user_id=slack_user_id,
                slack_to_openhands_user=slack_user,
                saas_user_auth=saas_user_auth,
                channel_id=channel_id,
                message_ts=message_ts,
                thread_ts=thread_ts,
                selected_repo=payload['selected_repo'],
                should_extract=True,
                send_summary_instruction=True,
                conversation_id='',
                team_id=team_id,
                v1_enabled=False,
            )

        else:
            return SlackNewConversationView(
                bot_access_token=bot_access_token,
                user_msg=user_msg,
                slack_user_id=slack_user_id,
                slack_to_openhands_user=slack_user,
                saas_user_auth=saas_user_auth,
                channel_id=channel_id,
                message_ts=message_ts,
                thread_ts=thread_ts,
                selected_repo=None,
                should_extract=True,
                send_summary_instruction=True,
                conversation_id='',
                team_id=team_id,
                v1_enabled=False,
            )
