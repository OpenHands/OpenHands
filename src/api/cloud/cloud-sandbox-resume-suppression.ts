const suppressedConversationIds = new Set<string>();

export function suppressNextCloudAutoResume(conversationId: string): void {
  suppressedConversationIds.add(conversationId);
}

export function consumeCloudAutoResumeSuppression(
  conversationId: string,
): boolean {
  return suppressedConversationIds.delete(conversationId);
}

export function __clearCloudAutoResumeSuppressionsForTests(): void {
  suppressedConversationIds.clear();
}
