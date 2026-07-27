import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  LLM_PROFILES_QUERY_KEYS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";
import { getLastRenderableEventId } from "#/hooks/chat/model-command-event-anchor";
import { recordModelSwitchMessage } from "#/hooks/chat/record-model-switch-message";
import {
  getStoredConversationMetadata,
  setStoredConversationMetadata,
} from "#/api/conversation-metadata-store";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { I18nKey } from "#/i18n/declaration";
import { invalidateConversationQueries } from "./conversation-mutation-utils";

interface SwitchLlmProfileVars {
  /**
   * When set, the conversation's running LLM is swapped via /switch_profile and
   * the user's global default profile is untouched. When null (home page),
   * the profile is activated globally instead.
   */
  conversationId: string | null;
  profileName: string;
}

/**
 * Shared key so any picker instance can observe an in-flight switch via
 * `useIsMutating` — the pill button and the menu that fires the switch are
 * separate hook instances, so per-observer `isPending` wouldn't line up.
 */
export const SWITCH_LLM_PROFILE_MUTATION_KEY = ["switch-llm-profile"];

/**
 * Switches the LLM profile. Per-conversation when called from inside a
 * conversation; globally activates the profile when called from the home page.
 *
 * The confirmation message, #1082 metadata persist, and error reporting all
 * live in the mutation-level callbacks (not `mutate(..., callbacks)`) so they
 * still run after the switcher menu closes on select — React Query drops
 * mutate-scoped callbacks when the calling component unmounts.
 *
 * `meta.disableToast` + a tailored `onError` (rather than the global mutation
 * toast) so a failed switch keeps the specific "Switched to {name} failed"
 * message instead of a generic error (#1571 review) — unlike
 * use-switch-acp-model, which has no per-action message worth tailoring.
 */
export const useSwitchLlmProfile = () => {
  const queryClient = useQueryClient();
  const { t } = useTranslation();

  return useMutation({
    mutationKey: SWITCH_LLM_PROFILE_MUTATION_KEY,
    mutationFn: ({ conversationId, profileName }: SwitchLlmProfileVars) =>
      AgentServerConversationService.switchProfile(conversationId, profileName),
    meta: { disableToast: true },
    // Anchor the inline message where the user issued the switch; captured
    // synchronously here because the menu unmounts before the switch resolves.
    onMutate: ({ conversationId }) => ({
      anchorEventId: conversationId ? getLastRenderableEventId() : null,
    }),
    onError: (error, { profileName }) => {
      const fallback = t(I18nKey.MODEL$SWITCH_FAILED, { name: profileName });
      displayErrorToast(retrieveAxiosErrorMessage(error) || fallback);
    },
    onSuccess: (_data, { conversationId, profileName }, context) => {
      queryClient.invalidateQueries({
        queryKey: LLM_PROFILES_QUERY_KEYS.all,
      });
      if (conversationId) {
        invalidateConversationQueries(queryClient, conversationId);
        recordModelSwitchMessage(
          conversationId,
          profileName,
          context?.anchorEventId ?? null,
        );
        // Keep the per-conversation profile identity fresh so the chat-header
        // switcher shows the right name after a reload (the agent-server only
        // round-trips the model string). #1082
        const prev = getStoredConversationMetadata(conversationId);
        setStoredConversationMetadata(conversationId, {
          selected_repository: prev?.selected_repository ?? null,
          selected_branch: prev?.selected_branch ?? null,
          git_provider: prev?.git_provider ?? null,
          selected_workspace: prev?.selected_workspace ?? null,
          active_profile: profileName,
          plugins: prev?.plugins ?? null,
        });
      } else {
        // Home-page activate path (same server endpoint as
        // useActivateLlmProfile): clear the SettingsService cache so the next
        // conversation-start reads the newly activated profile's LLM config
        // instead of the stale encrypted settings.
        SettingsService.invalidateCache();
        queryClient.invalidateQueries({
          queryKey: SETTINGS_QUERY_KEYS.personal(),
        });
      }
    },
  });
};
