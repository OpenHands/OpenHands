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

        if (action === "message") {
          const content: Array<MessageContent> = [];

          // Add text if present and non-empty (omits empty text blocks for image-only messages)
          if (args?.content && args.content.trim().length > 0) {
            content.push({
              type: "text",
              text: args.content,
            });
          }

          // Add images if present - using SDK's ImageContent format
          if (args?.image_urls && args.image_urls.length > 0) {
            content.push({
              type: "image",
              image_urls: args.image_urls,
            });
          }

          // Send only if there is at least one content block (text or image)
          if (content.length > 0) {
            const result = await conversationContext.sendMessage({
              role: "user",
              content,
            });
            return result;
          }
        }
        return { queued: false };
      }
      return { queued: false };
    },
    [conversationContext, conversationId],
  );

  return { send };
}
