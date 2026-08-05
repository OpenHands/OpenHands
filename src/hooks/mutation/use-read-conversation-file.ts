import { useMutation } from "@tanstack/react-query";
import { AxiosError } from "axios";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

interface UseReadConversationFileVariables {
  conversationId: string;
  filePath?: string;
}

const isNotFoundError = (error: unknown): boolean =>
  error instanceof AxiosError &&
  (error.response?.status === 404 || error.status === 404);

export const useReadConversationFile = () =>
  useMutation({
    mutationKey: ["read-conversation-file"],
    // Reading PLAN.md is an existence check that legitimately 404s when no
    // plan exists yet — that case is expected and handled locally by
    // callers, so it must stay silent. `disableToast` only suppresses the
    // generic MutationCache toast; the `onError` below re-shows one itself
    // for anything that isn't the expected 404, so real failures (a
    // transient 500, a network blip) still surface instead of leaving the
    // Planner tab silently stuck on its empty state.
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
      if (isNotFoundError(error)) return;
      displayErrorToast(retrieveAxiosErrorMessage(error));
    },
  });
