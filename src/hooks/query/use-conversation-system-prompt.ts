import { useQuery } from "@tanstack/react-query";
import EventService from "#/api/event-service/event-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import {
  adaptSystemMessage,
  SystemMessageForModal,
} from "#/utils/system-message-adapter";
import { isSystemPromptEvent } from "#/types/agent-server/type-guards";
import type { OpenHandsEvent } from "#/types/agent-server/core";

export const useConversationSystemPrompt = (
  conversationId: string | undefined,
  conversation: AppConversation | null | undefined,
  enabled: boolean = true,
) =>
  useQuery<SystemMessageForModal | null>({
    queryKey: [
      "conversation",
      conversationId,
      conversation?.conversation_url ?? null,
      conversation?.session_api_key ?? null,
      "system-prompt",
    ],
    queryFn: async () => {
      if (!conversationId) return null;
      try {
        const page = await EventService.searchEvents(
          conversationId,
          conversation?.conversation_url ?? null,
          conversation?.session_api_key ?? null,
          {
            limit: 50,
            sortOrder: "TIMESTAMP",
          },
        );
        const events = (page?.items ?? []) as OpenHandsEvent[];
        const systemPromptEvent = events.find(isSystemPromptEvent);
        if (!systemPromptEvent) return null;
        return adaptSystemMessage([systemPromptEvent]);
      } catch {
        return null;
      }
    },
    enabled: Boolean(enabled && conversationId),
    staleTime: 10 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    retry: 1,
  });
