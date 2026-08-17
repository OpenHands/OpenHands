import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { getLastRenderableEventId } from "#/hooks/chat/model-command-event-anchor";
import { buildSlashCommandItems } from "#/hooks/chat/use-slash-command";
import { useConversationSkills } from "#/hooks/query/use-conversation-skills";
import { useSlashCommandOutputStore } from "#/stores/slash-command-output-store";
import {
  HELP_COMMAND,
  FEEDBACK_COMMAND,
  FEEDBACK_FORM_URL,
} from "#/utils/constants";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";

/**
 * Intercepts browser-local utility commands. These commands never reach the
 * agent as user messages; help produces an anchored inline chat card and
 * feedback opens an external form.
 */
export function useSystemCommandInterceptor(
  conversationId: string | null | undefined,
  onSubmit: (message: string) => void,
) {
  const { t } = useTranslation("openhands");
  const { data: skills, refetch: refetchSkills } = useConversationSkills();
  const showHelp = useSlashCommandOutputStore((state) => state.showHelp);

  return useCallback(
    (message: string) => {
      const command = message.trim();
      const isSystemCommand = [HELP_COMMAND, FEEDBACK_COMMAND].includes(
        command,
      );
      if (!isSystemCommand) {
        onSubmit(message);
        return;
      }

      // @spec SC-004 — Feedback
      if (command === FEEDBACK_COMMAND) {
        window.open(FEEDBACK_FORM_URL, "_blank", "noopener,noreferrer");
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
      refetchSkills()
        .then((result) => {
          showHelp(
            conversationId,
            anchorEventId,
            buildSlashCommandItems(
              result.isError ? (skills ?? []) : (result.data ?? skills ?? []),
            ),
          );
        })
        .catch(() => displayErrorToast(t(I18nKey.ERROR$GENERIC)));
    },
    [conversationId, onSubmit, refetchSkills, showHelp, skills, t],
  );
}
