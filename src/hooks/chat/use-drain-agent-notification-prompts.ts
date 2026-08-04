import { useEffect, useRef } from "react";
import { useConversationWebSocket } from "#/contexts/conversation-websocket-context";
import {
  clearAgentNotificationPendingPrompts,
  readAgentNotificationPendingPrompts,
} from "#/components/features/chat/agent-notifications-pending-prompts";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";

export function useDrainAgentNotificationPrompts() {
  const { conversationId } = useOptionalConversationId();
  const conversationWebSocket = useConversationWebSocket();
  const drainedConversationIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (!conversationId || !conversationWebSocket?.sendMessage) {
      return;
    }

    if (drainedConversationIdRef.current === conversationId) {
      return;
    }

    const prompts = readAgentNotificationPendingPrompts(conversationId);
    if (prompts.length === 0) {
      return;
    }

    drainedConversationIdRef.current = conversationId;

    void (async () => {
      try {
        for (const prompt of prompts) {
          await conversationWebSocket.sendMessage({
            role: "user",
            content: [{ type: "text", text: prompt }],
          });
        }
      } finally {
        clearAgentNotificationPendingPrompts(conversationId);
      }
    })();
  }, [conversationId, conversationWebSocket]);
}
