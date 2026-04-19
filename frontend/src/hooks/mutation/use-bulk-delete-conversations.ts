import { useMutation, useQueryClient } from "@tanstack/react-query";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { clearConversationLocalStorage } from "#/utils/conversation-local-storage";
import {
  removeConversationsFromCache,
  restoreConversationsCache,
} from "./conversation-mutation-utils";

const BULK_DELETE_BATCH_SIZE = 50;

export const useBulkDeleteConversations = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationKey: ["bulk-delete-conversations"],
    mutationFn: async (variables: { conversationIds: string[] }) => {
      const batches: string[][] = [];
      for (
        let i = 0;
        i < variables.conversationIds.length;
        i += BULK_DELETE_BATCH_SIZE
      ) {
        batches.push(
          variables.conversationIds.slice(i, i + BULK_DELETE_BATCH_SIZE),
        );
      }

      const results = await batches.reduce(
        async (accPromise, batch) => {
          const acc = await accPromise;
          const result =
            await ConversationService.bulkDeleteConversations(batch);
          return {
            succeeded: [...acc.succeeded, ...result.succeeded],
            failed: [...acc.failed, ...result.failed],
          };
        },
        Promise.resolve({ succeeded: [] as string[], failed: [] as string[] }),
      );

      return results;
    },
    onMutate: async (variables) => {
      // Cancel any in-flight fetches and save snapshot for rollback
      await queryClient.cancelQueries({ queryKey: ["user", "conversations"] });
      const previousData = removeConversationsFromCache(
        queryClient,
        variables.conversationIds,
      );
      return { previousData };
    },
    onSuccess: (data) => {
      data.succeeded.forEach(clearConversationLocalStorage);
    },
    onError: (_err, _variables, context) => {
      if (context?.previousData) {
        restoreConversationsCache(queryClient, context.previousData);
      }
    },
    onSettled: async () => {
      await queryClient.invalidateQueries({
        queryKey: ["user", "conversations"],
      });
    },
  });
};
