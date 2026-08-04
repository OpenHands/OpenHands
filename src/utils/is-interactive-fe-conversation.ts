import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";

/**
 * Tag key automations / non-FE clients stamp for attribution
 * (e.g. ``origin=slack``). See conversation card tag-chip docs.
 */
export const CONVERSATION_ORIGIN_TAG_KEY = "origin";

type ConversationOriginFields = Pick<AppConversation, "trigger" | "tags">;

/**
 * True when this conversation was started interactively from the frontend
 * (or is otherwise safe for agent-notification suggestions).
 *
 * Non-interactive sources we skip:
 * - Cloud ``trigger`` values other than ``"gui"`` (resolver, suggested_task,
 *   microagent_management)
 * - Conversations with an ``origin`` tag (automation / external stamp)
 *
 * Local FE creates currently leave ``trigger`` null and omit ``origin``;
 * ACP-only ``acpserver`` tags do not disqualify.
 */
export function isInteractiveFeConversation(
  conversation: ConversationOriginFields | null | undefined,
): boolean {
  if (!conversation) {
    return false;
  }

  const { trigger, tags } = conversation;
  if (trigger != null && trigger !== "gui") {
    return false;
  }

  const origin = tags?.[CONVERSATION_ORIGIN_TAG_KEY];
  if (typeof origin === "string" && origin.trim() !== "") {
    return false;
  }

  return true;
}
