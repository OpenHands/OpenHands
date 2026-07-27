import { useInfiniteQuery } from "@tanstack/react-query";
import { SharedClient } from "@openhands/typescript-client/clients";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";

interface SharedEventPage {
  items: OpenHandsEvent[];
  next_page_id: string | null;
}

export const useSharedConversationEvents = (conversationId?: string) =>
  useInfiniteQuery({
    queryKey: ["shared-conversation-events", conversationId],
    queryFn: ({ pageParam }) => {
      if (!conversationId) {
        throw new Error("Conversation ID is required");
      }
      return new SharedClient(getAgentServerClientOptions()).searchSharedEvents(
        {
          conversationId,
          limit: 100,
          pageId: pageParam,
        },
      ) as Promise<SharedEventPage>;
    },
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_page_id ?? undefined,
    enabled: !!conversationId,
    retry: false, // Don't retry for shared conversations
  });
