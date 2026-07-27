import { useMutation, useQueryClient } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { invalidateConversationQueries } from "./conversation-mutation-utils";

interface SwitchAcpModelVars {
  /**
   * When set, the ACP conversation's running model is swapped live via the
   * wrapper's ``session/set_model`` (POST /switch_acp_model) and the user's
   * saved default is untouched. When null (home page / no session), the model
   * is persisted as the agent-settings default so the next conversation
   * created here inherits it.
   */
  conversationId: string | null;
  model: string;
}

/**
 * ACP analog of {@link useSwitchLlmProfile}. Switches the ACP model
 * per-conversation when called from inside a conversation (live in-place model
 * switch); persists it as the agent-settings default when called from the home
 * page (no live session to switch — the agent-server returns 409 before the
 * first message, so we write the default the next conversation inherits).
 *
 * Invalidates the same conversation/settings query keys the profile hook does
 * so the chat-input model chip + conversation chip refresh with the new model.
 */
export const useSwitchAcpModel = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ conversationId, model }: SwitchAcpModelVars) => {
      if (conversationId) {
        await AgentServerConversationService.switchAcpModel(
          conversationId,
          model,
        );
        return;
      }
      // Home page / no session: persist as the agent-settings default. The
      // backend deep-merges ``agent_settings_diff`` into the existing
      // ``agent_settings`` dict, so a scalar ``acp_model`` diff updates only
      // the model and preserves the selected provider + command.
      await SettingsService.saveSettings({
        agent_settings_diff: { acp_model: model },
      });
    },
    onSuccess: (_data, { conversationId }) => {
      if (conversationId) {
        invalidateConversationQueries(queryClient, conversationId);
      } else {
        // Mirror useSwitchLlmProfile's home-page path: clear the stale settings
        // cache so the next conversation-start reads the newly saved default,
        // and refetch the settings query so the home-page chip updates.
        SettingsService.invalidateCache();
        queryClient.invalidateQueries({
          queryKey: SETTINGS_QUERY_KEYS.personal(),
        });
      }
    },
    // No meta.disableToast: unlike useSwitchLlmProfile (which tailors a
    // "Switched to {name} failed" message via its own onError), there's no
    // per-action message worth tailoring here, so the global mutation error
    // toast reports a failed switch / settings write rather than swallowing it.
  });
};
