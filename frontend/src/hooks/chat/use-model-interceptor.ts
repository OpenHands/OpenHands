import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { useQueryClient } from "@tanstack/react-query";
import ProfilesService from "#/api/settings-service/profiles-service.api";
import { getRenderedV1Events } from "#/components/v1/chat";
import { useSwitchLlmProfileAndLog } from "#/hooks/mutation/use-switch-llm-profile-and-log";
import { LLM_PROFILES_QUERY_KEY } from "#/hooks/query/use-llm-profiles";
import { I18nKey } from "#/i18n/declaration";
import { useEventStore } from "#/stores/use-event-store";
import { useModelStore } from "#/stores/model-store";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { MODEL_COMMAND } from "#/utils/constants";

const MODEL_PREFIX = `${MODEL_COMMAND} `;

/**
 * Intercepts "/model" submissions:
 *   - "/model"        → render an inline list of saved profiles in the chat
 *   - "/model <name>" → switch the running conversation's LLM profile
 * Anything else (including /model on a V0 conversation) falls through.
 */
export const useModelInterceptor = (
  conversationId: string | null | undefined,
  onSubmit: (message: string) => void,
) => {
  const showProfiles = useModelStore((s) => s.show);
  const queryClient = useQueryClient();
  const { switchAndLog } = useSwitchLlmProfileAndLog();
  const { t } = useTranslation();

  return useCallback(
    (message: string) => {
      const trimmed = message.trim();
      const isModel =
        trimmed === MODEL_COMMAND || trimmed.startsWith(MODEL_PREFIX);
      if (!conversationId || !isModel) {
        onSubmit(message);
        return;
      }

      const arg = trimmed.slice(MODEL_COMMAND.length).trim();

      if (arg) {
        switchAndLog(conversationId, arg);
        return;
      }

      // Skip events filtered out by shouldRenderEvent (e.g.
      // ConversationStateUpdate) so the entry mounts where the user typed.
      const last = getRenderedV1Events(useEventStore.getState().uiEvents).at(
        -1,
      );
      const anchorEventId = last ? String(last.id) : null;

      // Imperative fetch through the query cache so the result populates the
      // same key `useLlmProfiles` reads, and a recently-fetched list is
      // reused. `staleTime: 0` forces a fresh fetch each time the user
      // explicitly asks via /model.
      queryClient
        .fetchQuery({
          queryKey: [LLM_PROFILES_QUERY_KEY],
          queryFn: ProfilesService.listProfiles,
          staleTime: 0,
        })
        .then(({ profiles }) =>
          showProfiles(conversationId, anchorEventId, profiles),
        )
        .catch((err) =>
          displayErrorToast(err?.message ?? t(I18nKey.MODEL$LIST_FAILED)),
        );
    },
    [conversationId, onSubmit, showProfiles, queryClient, switchAndLog, t],
  );
};
