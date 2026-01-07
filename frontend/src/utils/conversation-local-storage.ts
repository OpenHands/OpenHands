const CONVERSATION_STORAGE_KEYS = [
  "conversation-selected-tab-",
  "conversation-right-panel-shown-",
  "conversation-unpinned-tabs-",
];

export function clearConversationLocalStorage(conversationId: string) {
  try {
    CONVERSATION_STORAGE_KEYS.forEach((prefix) => {
      localStorage.removeItem(`${prefix}${conversationId}`);
    });
  } catch (err) {
    console.warn(
      "Failed to clear conversation localStorage",
      conversationId,
      err,
    );
  }
}

export function cleanupOrphanedConversationLocalStorage(
  existingConversationIds: string[],
) {
  try {
    const validIds = new Set(existingConversationIds);

    Object.keys(localStorage).forEach((key) => {
      const match = key.match(
        /^conversation-(selected-tab|right-panel-shown|unpinned-tabs)-(.+)$/,
      );

      if (!match) return;

      const conversationId = match[2];

      if (!validIds.has(conversationId)) {
        localStorage.removeItem(key);
      }
    });
  } catch (err) {
    console.warn("Failed to cleanup orphaned conversation localStorage", err);
  }
}
