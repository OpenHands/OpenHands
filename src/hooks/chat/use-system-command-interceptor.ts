import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { getLastRenderableEventId } from "#/hooks/chat/model-command-event-anchor";
import { useSlashCommandOutputStore } from "#/stores/slash-command-output-store";
import { BUILT_IN_COMMANDS, HELP_COMMAND } from "#/utils/constants";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";

/**
 * Intercepts browser-local utility commands. These commands never reach the
 * agent as user messages; help produces an anchored inline chat card.
 */
export function useSystemCommandInterceptor(
  conversationId: string | null | undefined,
  onSubmit: (message: string) => void,
) {
  const { t } = useTranslation("openhands");
  const showHelp = useSlashCommandOutputStore((state) => state.showHelp);

  return useCallback(
    (message: string) => {
      const command = message.trim();
      if (command !== HELP_COMMAND) {
        onSubmit(message);
        return;
      }

      if (!conversationId) {
        displayErrorToast(
          t(I18nKey.SLASH_COMMAND$ACTIVE_CONVERSATION_REQUIRED),
        );
        return;
      }

      // @spec SC-002 — Inline help
      const anchorEventId = getLastRenderableEventId();
      showHelp(conversationId, anchorEventId, BUILT_IN_COMMANDS);
    },
    [conversationId, onSubmit, showHelp, t],
  );
}
