import { useCallback } from "react";
import { useConversationWebSocket } from "#/contexts/conversation-websocket-context";
import { useConversationId } from "#/hooks/use-conversation-id";
import { V1MessageContent } from "#/api/conversation-service/v1-conversation-service.types";

interface SendResult {
  queued: boolean; // true if message was queued for later delivery
}

/**
 * Unified hook for sending messages that works with both V0 and V1 conversations
 * - For V0 conversations: Uses Socket.IO WebSocket via useWsClient
 * - For V1 conversations: Uses native WebSocket via ConversationWebSocketProvider
 */
export function useSendMessage() {
  const { conversationId } = useConversationId();

  // Get V1 context (will be null if not in V1 provider)
  const v1Context = useConversationWebSocket();

  const send = useCallback(
    async (event: Record<string, unknown>): Promise<SendResult> => {
      if (v1Context) {
        // V1: Convert V0 event format to V1 message format
        const { action, args } = event as {
          action: string;
          args?: {
            content?: string;
            image_urls?: string[];
            file_urls?: string[];
            timestamp?: string;
          };
        };

        if (action === "message" && args?.content) {
          // Build V1 message content array
          const content: Array<V1MessageContent> = [
            {
              type: "text",
              text: args.content,
            },
          ];

          // Add images if present - using SDK's ImageContent format
          if (args.image_urls && args.image_urls.length > 0) {
            content.push({
              type: "image",
              image_urls: args.image_urls,
            });
          }

          // Send via V1 WebSocket context (uses correct host/port)
          const result = await v1Context.sendMessage({
            role: "user",
            content,
          });
          return result;
        }
        return { queued: false };
      }
      return { queued: false };
    },
    [v1Context, conversationId],
  );

  return { send };
}
