import { InteractiveChatBox } from "#/components/features/chat/interactive-chat-box";
import { ConversationConfirmationButtons } from "#/components/shared/buttons/conversation-confirmation-buttons";

/**
 * Minimal chat surface used by popout isolation tests in place of the full
 * ChatInterface. Exercises the same scoped hooks (agent state, composer,
 * confirmation) without mounting the heavier conversation chrome.
 */
export function TestChatSurface() {
  return (
    <div data-testid="chat-interface">
      <ConversationConfirmationButtons />
      <InteractiveChatBox onSubmit={() => undefined} hasStartedConversation />
    </div>
  );
}
