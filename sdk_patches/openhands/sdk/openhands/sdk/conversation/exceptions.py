from openhands.sdk.conversation.types import ConversationID


ISSUE_URL = "https://github.com/OpenHands/software-agent-sdk/issues/new"


class ConversationRunError(RuntimeError):
    """Raised when a conversation run fails.

    Carries the conversation_id and persistence_dir to make resuming/debugging
    easier while preserving the original exception via exception chaining.
    """

    conversation_id: ConversationID
    persistence_dir: str | None
    original_exception: BaseException

    def __init__(
        self,
        conversation_id: ConversationID,
        original_exception: BaseException,
        persistence_dir: str | None = None,
        message: str | None = None,
    ) -> None:
        self.conversation_id = conversation_id
        self.persistence_dir = persistence_dir
        self.original_exception = original_exception
        default_msg = self._build_error_message(
            conversation_id, original_exception, persistence_dir
        )
        super().__init__(message or default_msg)

    @staticmethod
    def _build_error_message(
        conversation_id: ConversationID,
        original_exception: BaseException,
        persistence_dir: str | None,
    ) -> str:
        """Build a detailed error message with debugging information."""
        lines = [
            f"Conversation run failed for id={conversation_id}: {original_exception}",
        ]

        if persistence_dir:
            lines.append(f"\nConversation logs are stored at: {persistence_dir}")
            lines.append("\nTo help debug this issue, please file a bug report at:")
            lines.append(f"  {ISSUE_URL}")
            lines.append("and attach the conversation logs from the directory above.")

        return "\n".join(lines)
