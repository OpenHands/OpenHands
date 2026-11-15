import { useMutation, useQueryClient } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  persistHomeAcpSelection,
  refreshAfterAcpSelection,
} from "./acp-selection";

interface SwitchAcpModelVars {
  /**
   * When set, the ACP conversation's running model is swapped live via the
   * wrapper's ``session/set_model`` (POST /switch_acp_model) and the user's
   * saved default is untouched. When null (home page / no session), the model
   * is persisted as the default the next conversation launches with.
   */
  conversationId: string | null;
  /** Null saves no model, so the next conversation starts on the agent's own default. */
  model: string | null;
}

/**
 * ACP analog of {@link useSwitchLlmProfile}. Switches the ACP model
 * per-conversation when called from inside a conversation (live in-place model
 * switch; a created-but-not-yet-run conversation is a persist-only deferral on
 * the server, so blank conversations work too); persists it as the launch
 * default when called from the home page (see {@link persistHomeAcpSelection}).
 *
 * Invalidates the same conversation/settings query keys the profile hook does
 * so the chat-input model chip + conversation chip refresh with the new model.
 */
export const useSwitchAcpModel = () => {
  const queryClient = useQueryClient();
  const { backend, orgId } = useActiveBackend();

  return useMutation({
    mutationFn: async ({ conversationId, model }: SwitchAcpModelVars) => {
      if (conversationId) {
        if (model) {
          await AgentServerConversationService.switchAcpModel(
            conversationId,
            model,
          );
        }
        return;
      }

      await persistHomeAcpSelection(queryClient, backend.id, orgId, {
        acp_model: model,
      });
    },
    onSuccess: (_data, { conversationId }) =>
      refreshAfterAcpSelection(queryClient, conversationId),
    // No meta.disableToast: unlike useSwitchLlmProfile (which tailors a
    // "Switched to {name} failed" message via its own onError), there's no
    // per-action message worth tailoring here, so the global mutation error
    // toast reports a failed switch / profile or settings write rather than
    // swallowing it.
  });
};
