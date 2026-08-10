import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { getLastRenderableEventId } from "#/hooks/chat/model-command-event-anchor";
import { buildSlashCommandItems } from "#/hooks/chat/use-slash-command";
import { useConversationSkills } from "#/hooks/query/use-conversation-skills";
import { useSlashCommandOutputStore } from "#/stores/slash-command-output-store";
import { HELP_COMMAND, CONDENSE_COMMAND } from "#/utils/constants";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import { condenseConversation } from "#/hooks/mutation/conversation-mutation-utils";

const CONDENSE_UNSUPPORTED_STATUS_CODES = new Set([404, 405, 501]);
const CONDENSE_NOT_ENOUGH_HISTORY_PATTERNS = [
  "nocondensationavailableexception",
  "cannot condense 0 events",
  "unable to compute forgotten events",
  "events forgotten below minimum progress",
];

function getHttpStatus(error: unknown): number | undefined {
  if (typeof error !== "object" || error === null) return undefined;
  const directStatus = (error as { status?: unknown }).status;
  if (typeof directStatus === "number") return directStatus;
  const response = (error as { response?: unknown }).response;
  if (typeof response !== "object" || response === null) return undefined;
  const responseStatus = (response as { status?: unknown }).status;
  return typeof responseStatus === "number" ? responseStatus : undefined;
}

function isNotEnoughHistoryError(error: unknown): boolean {
  if (typeof error !== "object" || error === null) return false;

  const record = error as {
    message?: unknown;
    response?: unknown;
  };
  const response =
    typeof record.response === "object" && record.response !== null
      ? (record.response as { detail?: unknown; data?: unknown })
      : undefined;
  const responseData =
    typeof response?.data === "object" && response.data !== null
      ? (response.data as { detail?: unknown })
      : undefined;
  const details = [
    record.message,
    response?.detail,
    responseData?.detail,
    typeof record.response === "string" ? record.response : undefined,
  ]
    .filter((value): value is string => typeof value === "string")
    .join(" ")
    .toLowerCase();

  return CONDENSE_NOT_ENOUGH_HISTORY_PATTERNS.some((pattern) =>
    details.includes(pattern),
  );
}

/**
 * Intercepts browser-local utility commands. These commands never reach the
 * agent as user messages; help produces an anchored inline chat card and
 * condense triggers conversation history condensation.
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
      const isSystemCommand = [HELP_COMMAND, CONDENSE_COMMAND].includes(
        command,
      );
      if (!isSystemCommand) {
        onSubmit(message);
        return;
      }

      if (!conversationId) {
        displayErrorToast(
          t(I18nKey.SLASH_COMMAND$ACTIVE_CONVERSATION_REQUIRED),
        );
        return;
      }

      const anchorEventId = getLastRenderableEventId();

      // @spec SC-005 — Conversation condensation
      if (command === CONDENSE_COMMAND) {
        condenseConversation(conversationId)
          .then(() =>
            displaySuccessToast(t(I18nKey.SLASH_COMMAND$CONDENSE_SUCCESS)),
          )
          .catch((error: unknown) => {
            const status = getHttpStatus(error);
            let message = t(I18nKey.SLASH_COMMAND$CONDENSE_FAILED);
            if (isNotEnoughHistoryError(error)) {
              message = t(I18nKey.SLASH_COMMAND$CONDENSE_NOT_ENOUGH_HISTORY);
            } else if (
              status !== undefined &&
              CONDENSE_UNSUPPORTED_STATUS_CODES.has(status)
            ) {
              message = t(I18nKey.SLASH_COMMAND$CONDENSE_UNSUPPORTED);
            }
            displayErrorToast(message);
          });
        return;
      }

      // @spec SC-002 — Inline help
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
