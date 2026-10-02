import { useInfiniteQuery } from "@tanstack/react-query";
import { SharedClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import {
  searchCloudSharedEvents,
  type SharedEventPage,
} from "#/api/cloud/shared-conversation-service.api";

export const useSharedConversationEvents = (conversationId?: string) =>
  useInfiniteQuery({
    queryKey: ["shared-conversation-events", conversationId],
    queryFn: ({ pageParam }): Promise<SharedEventPage> => {
      if (!conversationId) {
        throw new Error("Conversation ID is required");
      }
      const request = { conversationId, limit: 100, pageId: pageParam };
      if (getActiveBackend().backend.kind === "cloud") {
        return searchCloudSharedEvents(request);
      }
      return new SharedClient(getAgentServerClientOptions()).searchSharedEvents(
        request,
      ) as Promise<SharedEventPage>;
    },
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_page_id ?? undefined,
    enabled: !!conversationId,
    retry: false, // Don't retry for shared conversations
  });
