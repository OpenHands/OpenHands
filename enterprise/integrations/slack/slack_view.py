from dataclasses import dataclass

from integrations.models import Message
from integrations.slack.slack_types import SlackViewInterface, StartingConvoException
from integrations.utils import CONVERSATION_URL, get_final_agent_observation
from jinja2 import Environment
from slack_sdk import WebClient
from storage.slack_conversation import SlackConversation
from storage.slack_conversation_store import SlackConversationStore
from storage.slack_team_store import SlackTeamStore
from storage.slack_user import SlackUser

from openhands.core.logger import openhands_logger as logger
from openhands.core.schema.agent import AgentState
from openhands.events.action import MessageAction
from openhands.events.serialization.event import event_to_dict
from openhands.integrations.provider import ProviderHandler
from openhands.server.services.conversation_service import (
    create_new_conversation,
    setup_init_conversation_settings,
)
from openhands.server.shared import ConversationStoreImpl, config, conversation_manager
from openhands.server.user_auth.user_auth import UserAuth
from openhands.storage.data_models.conversation_metadata import ConversationTrigger
from openhands.utils.async_utils import GENERAL_TIMEOUT, call_async_from_sync

# =================================================
# SECTION: Github view types
# =================================================


CONTEXT_LIMIT = 21
slack_conversation_store = SlackConversationStore.get_instance()
slack_team_store = SlackTeamStore.get_instance()


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

    def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        raise NotImplementedError

    async def create_or_update_conversation(self, jinja_env: Environment):
        raise NotImplementedError

    def get_callback_id(self) -> str:
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
        "Instructions passed when conversation is first initialized"

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

    async def save_slack_convo(self):
        if self.slack_to_openhands_user:
            user_info: SlackUser = self.slack_to_openhands_user

            logger.info(
                'Create slack conversation',
                extra={
                    'channel_id': self.channel_id,
                    'conversation_id': self.conversation_id,
                    'keycloak_user_id': user_info.keycloak_user_id,
                    'parent_id': self.thread_ts or self.message_ts,
                },
            )
            slack_conversation = SlackConversation(
                conversation_id=self.conversation_id,
                channel_id=self.channel_id,
                keycloak_user_id=user_info.keycloak_user_id,
                parent_id=self.thread_ts
                or self.message_ts,  # conversations can start in a thread reply as well; we should always references the parent's (root level msg's) message ID
            )
            await slack_conversation_store.create_slack_conversation(slack_conversation)

    async def create_or_update_conversation(self, jinja: Environment) -> str:
        """
        Only creates a new conversation
        """
        self._verify_necessary_values_are_set()

        provider_tokens = await self.saas_user_auth.get_provider_tokens()
        user_secrets = await self.saas_user_auth.get_secrets()
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
        await self.save_slack_convo()
        return self.conversation_id

    def get_callback_id(self) -> str:
        return f'slack_{self.channel_id}_{self.message_ts}'

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

    async def create_or_update_conversation(self, jinja: Environment) -> str:
        """
        Send new user message to converation
        """
        user_info: SlackUser = self.slack_to_openhands_user
        saas_user_auth: UserAuth = self.saas_user_auth
        user_id = user_info.keycloak_user_id

        # Org management in the future will get rid of this
        # For now, only user that created the conversation can send follow up messages to it
        if user_id != self.slack_conversation.keycloak_user_id:
            raise StartingConvoException(
                f'{user_info.slack_display_name} is not authorized to send messages to this conversation.'
            )

        # Check if conversation has been deleted
        # Update logic when soft delete is implemented
        conversation_store = await ConversationStoreImpl.get_instance(config, user_id)

        try:
            await conversation_store.get_metadata(self.conversation_id)
        except FileNotFoundError:
            raise StartingConvoException('Conversation no longer exists.')

        provider_tokens = await saas_user_auth.get_provider_tokens()

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

        user_msg, _ = self._get_instructions(jinja)
        user_msg_action = MessageAction(content=user_msg)
        await conversation_manager.send_event_to_conversation(
            self.conversation_id, event_to_dict(user_msg_action)
        )

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
            raise Exception('Did not slack team')

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
            )
