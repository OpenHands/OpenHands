import { useMutation, useQueryClient } from "@tanstack/react-query";
import toast from "react-hot-toast";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useStopConversation = () => {
  const { organizationId } = useSelectedOrganizationId();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ conversationId }: { conversationId: string }) =>
      organizationService.stopConversation({
        orgId: organizationId!,
        conversationId,
      }),
    onError: () => {
      toast.error("Failed to stop conversation");
    },
    onSuccess: () => {
      toast.success("Conversation stopped");
      queryClient.invalidateQueries({
        queryKey: ["organizations", "conversations", organizationId],
      });
      queryClient.invalidateQueries({
        queryKey: ["organizations", "conversation-stats", organizationId],
      });
    },
  });
};
