import { useQuery } from "@tanstack/react-query";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";

export const useConversationsInSandbox = (sandboxId: string | null) =>
  useQuery({
    queryKey: ["conversations", "sandbox", sandboxId],
    queryFn: () =>
      sandboxId
        ? V1ConversationService.searchConversationsBySandboxId(sandboxId)
        : Promise.resolve([]),
    enabled: !!sandboxId,
    staleTime: 0, // Always consider data stale for confirmation dialogs
    gcTime: 1000 * 60, // 1 minute
    refetchOnMount: true, // Always fetch fresh data when modal opens
  });
