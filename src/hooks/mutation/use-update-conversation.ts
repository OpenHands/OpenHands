import { assertActiveEnvironment } from "#/utils/assert-active-environment";
import {
  useMutation,
  useQueryClient,
  type InfiniteData,
} from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type {
  AppConversation,
  AppConversationPage,
} from "#/api/conversation-service/agent-server-conversation-service.types";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  CONVERSATION_QUERY_KEYS,
  DEFAULT_CONVERSATION_PAGE_SIZE,
} from "../query/query-keys";

export const useUpdateConversation = () => {
  const queryClient = useQueryClient();
  const { backend, orgId } = useActiveBackend();
  const listKey = CONVERSATION_QUERY_KEYS.paginated(
    DEFAULT_CONVERSATION_PAGE_SIZE,
    backend.id,
    orgId,
  );
  const detailKey = (id: string) =>
    CONVERSATION_QUERY_KEYS.detail(id, backend.id, orgId);

  return useMutation({
    mutationFn: (variables: { conversationId: string; newTitle: string }) => {
      assertActiveEnvironment({ backend, orgId });
      return AgentServerConversationService.updateConversationTitle(
        variables.conversationId,
        variables.newTitle,
      );
    },
    onMutate: async ({ conversationId, newTitle }) => {
      await Promise.all([
        queryClient.cancelQueries({ queryKey: listKey, exact: true }),
        queryClient.cancelQueries({
          queryKey: detailKey(conversationId),
          exact: true,
        }),
      ]);
      const previousList =
        queryClient.getQueryData<InfiniteData<AppConversationPage>>(listKey);
      const previousDetail = queryClient.getQueryData<AppConversation>(
        detailKey(conversationId),
      );
      const rename = (conversation: AppConversation) =>
        conversation.id === conversationId
          ? {
              ...conversation,
              title: newTitle,
              title_generation: null,
              updated_at: new Date().toISOString(),
            }
          : conversation;
      queryClient.setQueryData<InfiniteData<AppConversationPage>>(
        listKey,
        (old) =>
          old
            ? {
                ...old,
                pages: old.pages.map((page) => ({
                  ...page,
                  items: page.items.map(rename),
                })),
              }
            : old,
      );
      queryClient.setQueryData<AppConversation>(
        detailKey(conversationId),
        (old) => (old ? rename(old) : old),
      );
      return { previousList, previousDetail };
    },
    onError: (_error, variables, context) => {
      if (context?.previousList)
        queryClient.setQueryData(listKey, context.previousList);
      if (context?.previousDetail)
        queryClient.setQueryData(
          detailKey(variables.conversationId),
          context.previousDetail,
        );
    },
    onSettled: (_data, _error, variables) => {
      void queryClient.invalidateQueries({ queryKey: listKey, exact: true });
      void queryClient.invalidateQueries({
        queryKey: detailKey(variables.conversationId),
        exact: true,
      });
    },
  });
};
