/**
 * While a cloud start task redirects from `/conversations/task-{uuid}` to the
 * real conversation id, optimistic pending messages stay keyed to the task id
 * until a layout effect can reassign them. This map lets the chat UI keep
 * matching those bubbles against the real conversation route.
 */
const taskSourceByRealConversationId = new Map<string, string>();

let scheduledPendingReassign: {
  fromConversationId: string;
  toConversationId: string;
} | null = null;

export function linkPendingTaskMessages(
  realConversationId: string,
  taskConversationId: string,
): void {
  taskSourceByRealConversationId.set(realConversationId, taskConversationId);
}

export function clearPendingTaskMessageLink(realConversationId: string): void {
  taskSourceByRealConversationId.delete(realConversationId);
}

export function schedulePendingTaskMessageReassign(
  fromConversationId: string,
  toConversationId: string,
): void {
  scheduledPendingReassign = { fromConversationId, toConversationId };
}

export function consumeScheduledPendingTaskMessageReassign(
  conversationId: string,
): { fromConversationId: string; toConversationId: string } | null {
  if (scheduledPendingReassign?.toConversationId !== conversationId) {
    return null;
  }

  const value = scheduledPendingReassign;
  scheduledPendingReassign = null;
  return value;
}

export function matchesPendingConversationId(
  activeConversationId: string,
  pendingConversationId: string,
): boolean {
  if (pendingConversationId === activeConversationId) {
    return true;
  }

  const linkedTaskConversationId =
    taskSourceByRealConversationId.get(activeConversationId);
  return linkedTaskConversationId === pendingConversationId;
}

/** Test helper */
export function resetPendingTaskMessageLinkState(): void {
  taskSourceByRealConversationId.clear();
  scheduledPendingReassign = null;
}
