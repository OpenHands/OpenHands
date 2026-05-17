import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { useSwitchLlmProfile } from "#/hooks/mutation/use-switch-llm-profile";
import { useModelStore } from "#/stores/model-store";
import { useEventStore } from "#/stores/use-event-store";
import { getRenderedV1Events } from "#/components/v1/chat";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { extractErrorMessage } from "#/utils/extract-error-message";
import { I18nKey } from "#/i18n/declaration";

/** Switch the conversation's LLM profile and render the result inline (same UX as `/model <name>`). */
export function useSwitchLlmProfileAndLog() {
  const { mutate, isPending } = useSwitchLlmProfile();
  const recordSwitch = useModelStore((s) => s.recordSwitch);
  const { t } = useTranslation();

  // Stable identity so the /model interceptor's outer useCallback doesn't bust each render.
  const switchAndLog = useCallback(
    (conversationId: string, profileName: string) => {
      const last = getRenderedV1Events(useEventStore.getState().uiEvents).at(
        -1,
      );
      const anchorEventId = last ? String(last.id) : null;

      mutate(
        { conversationId, profileName },
        {
          onSuccess: () =>
            recordSwitch(conversationId, anchorEventId, profileName),
          onError: (err) =>
            displayErrorToast(
              extractErrorMessage(
                err,
                t(I18nKey.MODEL$SWITCH_FAILED, { name: profileName }),
              ),
            ),
        },
      );
    },
    [mutate, recordSwitch, t],
  );

  return { switchAndLog, isPending };
}
