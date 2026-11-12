import { I18nKey } from "#/i18n/declaration";
import type { ProjectConversation } from "#/api/projects-service/projects-service.api";

const BY_STATUS: Record<string, I18nKey> = {
  waiting_for_confirmation: I18nKey.PROJECTS$STATE_WAITING_APPROVAL,
  error: I18nKey.PROJECTS$STATE_ERROR,
  stuck: I18nKey.PROJECTS$STATE_STUCK,
  paused: I18nKey.PROJECTS$STATE_PAUSED,
  running: I18nKey.PROJECTS$GROUP_WORKING,
};

/**
 * What a project conversation is doing, in the user's words: which of the
 * things under Needs you it is (an approval, a question, a failure), not the
 * server's status string.
 */
export function conversationStateLabel(
  conversation: Pick<ProjectConversation, "execution_status" | "turn">,
): I18nKey {
  const byStatus = BY_STATUS[conversation.execution_status];
  if (byStatus) return byStatus;
  if (conversation.turn === "asked") return I18nKey.PROJECTS$STATE_ASKED;
  return conversation.execution_status === "finished"
    ? I18nKey.PROJECTS$STATE_FINISHED
    : I18nKey.PROJECTS$GROUP_IDLE;
}

export const needsUser = (
  conversation: Pick<ProjectConversation, "attention">,
) =>
  conversation.attention === "needs_you" ||
  conversation.attention === "needs_input";
