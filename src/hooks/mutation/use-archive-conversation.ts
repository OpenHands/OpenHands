import { useMutation, useQueryClient } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { ARCHIVED_CONVERSATION_TAG_KEY } from "#/api/agent-server-adapter";
import { patchConversationInCache } from "./conversation-mutation-utils";

export type ArchiveConversationVariables = {
  conversationId: string;
  archived: boolean;
  currentTags: Record<string, string> | null | undefined;
};

export const useArchiveConversation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      conversationId,
      archived,
      currentTags,
    }: ArchiveConversationVariables) => {
      const nextTags = buildNextTags(currentTags, archived);
      return AgentServerConversationService.updateConversationTags(
        conversationId,
        nextTags,
      );
    },
    onSuccess: (_, { conversationId, archived, currentTags }) => {
      const nextTags = buildNextTags(currentTags, archived);
      patchConversationInCache(queryClient, conversationId, { tags: nextTags });
    },
    onError: () => {
      // Invalidate so the list re-fetches and shows accurate state.
      queryClient.invalidateQueries({ queryKey: ["user", "conversations"] });
    },
  });
};

function buildNextTags(
  currentTags: Record<string, string> | null | undefined,
  archived: boolean,
): Record<string, string> {
  const base = currentTags ?? {};
  if (archived) {
    return { ...base, [ARCHIVED_CONVERSATION_TAG_KEY]: "true" };
  }
  return Object.fromEntries(
    Object.entries(base).filter(([k]) => k !== ARCHIVED_CONVERSATION_TAG_KEY),
  );
}
