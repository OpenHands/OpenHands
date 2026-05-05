import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { useQueryClient } from "@tanstack/react-query";
import ProfilesService from "#/api/settings-service/profiles-service.api";
import { getRenderedV1Events } from "#/components/v1/chat";
import { useSwitchLlmProfile } from "#/hooks/mutation/use-switch-llm-profile";
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
  const recordSwitch = useModelStore((s) => s.recordSwitch);
  const queryClient = useQueryClient();
  const { mutate: switchLlmProfile } = useSwitchLlmProfile();
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

      // Anchor entries to the latest v1 event so they render inline at the
      // chat position where the user typed /model, instead of always at the
      // bottom of the chat history. Use the shared `getRenderedV1Events` so
      // we only anchor to events that are actually rendered — anchoring to a
      // hidden event (e.g. ConversationStateUpdate) would leave the entry
      // with no slot to mount in.
      const renderedEvents = getRenderedV1Events(
        useEventStore.getState().uiEvents,
      );
      const anchorEventId =
        renderedEvents.length > 0
          ? String(renderedEvents[renderedEvents.length - 1].id)
          : null;

      if (!arg) {
        // Imperative fetch through the query cache so the result populates
        // the same key `useLlmProfiles` reads, and a recently-fetched list
        // is reused. `staleTime: 0` forces a fresh fetch each time the user
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
        return;
      }

      switchLlmProfile(
        { conversationId, profileName: arg },
        {
          onSuccess: () => recordSwitch(conversationId, anchorEventId, arg),
          onError: (err: unknown) => {
            const e = err as
              | { response?: { data?: { detail?: string } }; message?: string }
              | undefined;
            displayErrorToast(
              e?.response?.data?.detail ??
                e?.message ??
                t(I18nKey.MODEL$SWITCH_FAILED, { name: arg }),
            );
          },
        },
      );
    },
    [
      conversationId,
      onSubmit,
      showProfiles,
      recordSwitch,
      queryClient,
      switchLlmProfile,
      t,
    ],
  );
};
