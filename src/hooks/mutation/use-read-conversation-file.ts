import { useMutation } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { getErrorStatus } from "#/hooks/query/use-settings";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

interface UseReadConversationFileVariables {
  conversationId: string;
  filePath?: string;
}

export const useReadConversationFile = () =>
  useMutation({
    mutationKey: ["read-conversation-file"],
    // A 404 (no plan yet) is expected and must stay silent; `onError` below
    // re-shows the toast for anything else so real failures still surface.
    meta: { disableToast: true },
    mutationFn: async ({
      conversationId,
      filePath,
    }: UseReadConversationFileVariables): Promise<string> =>
      AgentServerConversationService.readConversationFile(
        conversationId,
        filePath,
      ),
    onError: (error) => {
      if (getErrorStatus(error) === 404) return;
      displayErrorToast(retrieveAxiosErrorMessage(error));
    },
  });
