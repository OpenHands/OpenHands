import { useUserConversation } from "#/hooks/query/use-user-conversation";
import { ConversationStatusDot } from "#/components/features/conversation-panel/conversation-status-dot";

interface PopoutStatusDotProps {
  conversationId: string;
  showTooltip?: boolean;
}

/**
 * Agent status for a docked popout, shown in its title bar.
 *
 * Reads through `useUserConversation`, which the popout's own
 * `WebSocketProviderWrapper` already keeps warm under the same query key — so
 * this shares that cache entry instead of opening a second polling loop. It
 * cannot reuse `ConversationNameWithStatus`: that one derives status from the
 * process-wide conversation-state store, which popouts deliberately
 * leave to the primary routed conversation.
 */
export function PopoutStatusDot({
  conversationId,
  showTooltip = true,
}: PopoutStatusDotProps) {
  const { data: conversation } = useUserConversation(conversationId);

  return (
    <ConversationStatusDot
      executionStatus={conversation?.execution_status ?? null}
      sandboxStatus={conversation?.sandbox_status ?? null}
      showTooltip={showTooltip}
    />
  );
}
