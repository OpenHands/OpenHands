import { useMutation, useQueryClient } from "@tanstack/react-query";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { clearConversationLocalStorage } from "#/utils/conversation-local-storage";

export const useDeleteConversations = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (variables: { conversationIds: string[] }) => {
      const results = await Promise.allSettled(
        variables.conversationIds.map((id) =>
          ConversationService.deleteUserConversation(id),
        ),
      );

      const failures = results.filter((r) => r.status === "rejected");
      if (failures.length > 0) {
        throw new Error(
          `Failed to delete ${failures.length} of ${variables.conversationIds.length} conversations`,
        );
      }
    },
    onMutate: async (variables) => {
      await queryClient.cancelQueries({
        queryKey: ["user", "conversations"],
      });

      const idsToDelete = new Set(variables.conversationIds);
      queryClient.setQueriesData<{
        pages: Array<{
          results: Array<{ conversation_id: string }>;
          next_page_id?: string;
        }>;
      }>({ queryKey: ["user", "conversations"] }, (oldData) => {
        if (!oldData) return oldData;
        return {
          ...oldData,
          pages: oldData.pages.map((page) => ({
            ...page,
            results: page.results.filter(
              (conv) => !idsToDelete.has(conv.conversation_id),
            ),
          })),
        };
      });
    },

    onSuccess: (_, variables) => {
      for (const id of variables.conversationIds) {
        clearConversationLocalStorage(id);
      }
    },

    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["user", "conversations"] });
    },
  });
};
