import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { isExecutionPaused } from "./status";
import { isRuntimeUnavailableSandboxStatus } from "./conversation-archive-status";

export type WakeableConversation = Pick<
  AppConversation,
  "id" | "execution_status" | "sandbox_id" | "sandbox_status"
>;

/**
 * A conversation is asleep when opening it would wake it: a paused cloud
 * sandbox (the conversation route auto-resumes it) or a paused execution
 * (the status menu's "Start server" action). Gone/errored runtimes cannot be
 * woken at all.
 */
export function isWakeableConversation(
  conversation: WakeableConversation,
): boolean {
  if (isRuntimeUnavailableSandboxStatus(conversation.sandbox_status)) {
    return false;
  }
  return (
    conversation.sandbox_status === "PAUSED" ||
    isExecutionPaused(conversation.execution_status)
  );
}

/**
 * The conversations a "wake all" bulk action should resume: every asleep
 * conversation that is not archived. Archived rows are excluded even when
 * asleep — the user put them away, and waking an archived runtime bills for
 * a sandbox nobody is about to use.
 */
export function collectWakeableConversations(
  conversations: readonly WakeableConversation[],
  isArchived: (conversation: WakeableConversation) => boolean,
): WakeableConversation[] {
  return conversations.filter(
    (conversation) =>
      !isArchived(conversation) && isWakeableConversation(conversation),
  );
}
