import { useCallback } from "react";
import { useConversationWebSocket } from "#/contexts/conversation-websocket-context";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { MessageContent } from "#/api/conversation-service/agent-server-conversation-service.types";

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

        const text = args?.content;
        const imageUrls = args?.image_urls ?? [];

        if (action === "message" && (text || imageUrls.length > 0)) {
          // Build agent-server message content array
          const content: Array<MessageContent> = [];

          // An empty text block makes the model treat the message as empty and ignore the image
          if (text) {
            content.push({
              type: "text",
              text,
            });
          }

          // Add images if present - using SDK's ImageContent format
          if (imageUrls.length > 0) {
            content.push({
              type: "image",
              image_urls: imageUrls,
            });
          }

          // Send via WebSocket context (uses correct host/port)
          const result = await conversationContext.sendMessage({
            role: "user",
            content,
          });
          return result;
        }
        return { queued: false };
      }
      return { queued: false };
    },
    [conversationContext, conversationId],
  );

  return { send };
}
