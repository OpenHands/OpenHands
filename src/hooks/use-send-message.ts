import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { useConversationWebSocket } from "#/contexts/conversation-websocket-context";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { buildUserMessageContent } from "#/utils/send-message-content";

interface SendResult {
  queued: boolean; // true if message was queued for later delivery
}

/**
 * Sends user messages through the active conversation WebSocket.
 */
export function useSendMessage() {
  // Optional: this hook is reachable from the home-page chat input shell.
  // Outside a conversation route, `conversationContext` is null anyway so
  // `send` is a no-op that returns `{ queued: false }`.
  const { conversationId } = useOptionalConversationId();

  // Get agent-server context (null outside a conversation provider)
  const conversationContext = useConversationWebSocket();
  const { t } = useTranslation("openhands");

  const send = useCallback(
    async (event: Record<string, unknown>): Promise<SendResult> => {
      if (conversationContext) {
        // Convert chat input payloads to agent-server message content.
        const { action, args } = event as {
          action: string;
          args?: {
            content?: string;
            image_urls?: string[];
            file_urls?: string[];
            timestamp?: string;
          };
        };

        const hasAttachments =
          (args?.image_urls?.length ?? 0) > 0 ||
          (args?.file_urls?.length ?? 0) > 0;

        if (action === "message" && (args?.content || hasAttachments)) {
          // The shared builder keeps the WebSocket payload identical to the
          // REST path's, including uploaded-file references (previously
          // typed but never forwarded here).
          const content = buildUserMessageContent({
            text: args?.content ?? "",
            imageUrls: args?.image_urls,
            fileUrls: args?.file_urls,
            t,
          });

          // Send via WebSocket context (uses correct host/port)
          return await conversationContext.sendMessage({
            role: "user",
            content,
          });
        }
        return { queued: false };
      }
      return { queued: false };
    },
    [conversationContext, conversationId, t],
  );

  return { send };
}
