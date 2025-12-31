from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Self, overload

from openhands.sdk.agent.base import AgentBase
from openhands.sdk.conversation.base import BaseConversation
from openhands.sdk.conversation.types import (
    ConversationCallbackType,
    ConversationID,
    ConversationTokenCallbackType,
    StuckDetectionThresholds,
)
from openhands.sdk.conversation.visualizer import (
    ConversationVisualizerBase,
    DefaultConversationVisualizer,
)
from openhands.sdk.hooks import HookConfig
from openhands.sdk.logger import get_logger
from openhands.sdk.secret import SecretValue
from openhands.sdk.workspace import LocalWorkspace, RemoteWorkspace


if TYPE_CHECKING:
    from openhands.sdk.conversation.impl.local_conversation import LocalConversation
    from openhands.sdk.conversation.impl.remote_conversation import RemoteConversation

logger = get_logger(__name__)


class Conversation:
    """Factory class for creating conversation instances with OpenHands agents.

    This factory automatically creates either a LocalConversation or RemoteConversation
    based on the workspace type provided. LocalConversation runs the agent locally,
    while RemoteConversation connects to a remote agent server.

    Returns:
        LocalConversation if workspace is local, RemoteConversation if workspace
        is remote.

    Example:
        >>> from openhands.sdk import LLM, Agent, Conversation
        >>> llm = LLM(model="claude-sonnet-4-20250514", api_key=SecretStr("key"))
        >>> agent = Agent(llm=llm, tools=[])
        >>> conversation = Conversation(agent=agent, workspace="./workspace")
        >>> conversation.send_message("Hello!")
        >>> conversation.run()
    """

    @overload
    def __new__(
        cls: type[Self],
        agent: AgentBase,
        *,
        workspace: str | Path | LocalWorkspace = "workspace/project",
        persistence_dir: str | Path | None = None,
        conversation_id: ConversationID | None = None,
        callbacks: list[ConversationCallbackType] | None = None,
        token_callbacks: list[ConversationTokenCallbackType] | None = None,
        hook_config: HookConfig | None = None,
        max_iteration_per_run: int = 500,
        stuck_detection: bool = True,
        stuck_detection_thresholds: (
            StuckDetectionThresholds | Mapping[str, int] | None
        ) = None,
        visualizer: (
            type[ConversationVisualizerBase] | ConversationVisualizerBase | None
        ) = DefaultConversationVisualizer,
        secrets: dict[str, SecretValue] | dict[str, str] | None = None,
    ) -> "LocalConversation": ...

    @overload
    def __new__(
        cls: type[Self],
        agent: AgentBase,
        *,
        workspace: RemoteWorkspace,
        conversation_id: ConversationID | None = None,
        callbacks: list[ConversationCallbackType] | None = None,
        token_callbacks: list[ConversationTokenCallbackType] | None = None,
        hook_config: HookConfig | None = None,
        max_iteration_per_run: int = 500,
        stuck_detection: bool = True,
        stuck_detection_thresholds: (
            StuckDetectionThresholds | Mapping[str, int] | None
        ) = None,
        visualizer: (
            type[ConversationVisualizerBase] | ConversationVisualizerBase | None
        ) = DefaultConversationVisualizer,
        secrets: dict[str, SecretValue] | dict[str, str] | None = None,
    ) -> "RemoteConversation": ...

    def __new__(
        cls: type[Self],
        agent: AgentBase,
        *,
        workspace: str | Path | LocalWorkspace | RemoteWorkspace = "workspace/project",
        persistence_dir: str | Path | None = None,
        conversation_id: ConversationID | None = None,
        callbacks: list[ConversationCallbackType] | None = None,
        token_callbacks: list[ConversationTokenCallbackType] | None = None,
        hook_config: HookConfig | None = None,
        max_iteration_per_run: int = 500,
        stuck_detection: bool = True,
        stuck_detection_thresholds: (
            StuckDetectionThresholds | Mapping[str, int] | None
        ) = None,
        visualizer: (
            type[ConversationVisualizerBase] | ConversationVisualizerBase | None
        ) = DefaultConversationVisualizer,
        secrets: dict[str, SecretValue] | dict[str, str] | None = None,
    ) -> BaseConversation:
        from openhands.sdk.conversation.impl.local_conversation import LocalConversation
        from openhands.sdk.conversation.impl.remote_conversation import (
            RemoteConversation,
        )

        if isinstance(workspace, RemoteWorkspace):
            # For RemoteConversation, persistence_dir should not be used
            # Only check if it was explicitly set to something other than the default
            if persistence_dir is not None:
                raise ValueError(
                    "persistence_dir should not be set when using RemoteConversation"
                )
            return RemoteConversation(
                agent=agent,
                conversation_id=conversation_id,
                callbacks=callbacks,
                token_callbacks=token_callbacks,
                hook_config=hook_config,
                max_iteration_per_run=max_iteration_per_run,
                stuck_detection=stuck_detection,
                stuck_detection_thresholds=stuck_detection_thresholds,
                visualizer=visualizer,
                workspace=workspace,
                secrets=secrets,
            )

        return LocalConversation(
            agent=agent,
            conversation_id=conversation_id,
            callbacks=callbacks,
            token_callbacks=token_callbacks,
            hook_config=hook_config,
            max_iteration_per_run=max_iteration_per_run,
            stuck_detection=stuck_detection,
            stuck_detection_thresholds=stuck_detection_thresholds,
            visualizer=visualizer,
            workspace=workspace,
            persistence_dir=persistence_dir,
            secrets=secrets,
        )
