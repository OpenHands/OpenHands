export const CONVERSATION_STORAGE_KEYS = {
  selectedTab: (conversationId: string) =>
    `conversation-selected-tab-${conversationId}`,
  rightPanelShown: (conversationId: string) =>
    `conversation-right-panel-shown-${conversationId}`,
  unpinnedTabs: (conversationId: string) =>
    `conversation-unpinned-tabs-${conversationId}`,
};

export const clearConversationStorage = (conversationId: string): void => {
  try {
    localStorage.removeItem(
      CONVERSATION_STORAGE_KEYS.selectedTab(conversationId),
    );
    localStorage.removeItem(
      CONVERSATION_STORAGE_KEYS.rightPanelShown(conversationId),
    );
    localStorage.removeItem(
      CONVERSATION_STORAGE_KEYS.unpinnedTabs(conversationId),
    );
  } catch {
    // ignore
  }
};

export const findStoredConversationIds = (): string[] => {
  const conversationIds: string[] = [];
  const keyPattern =
    /^conversation-(?:selected-tab|right-panel-shown|unpinned-tabs)-(.+)$/;

  try {
    for (let index = 0; index < localStorage.length; index += 1) {
      const key = localStorage.key(index);
      if (key) {
        const match = key.match(keyPattern);
        if (match?.[1]) {
          conversationIds.push(match[1]);
        }
      }
    }
  } catch {
    // ignore
  }

  return Array.from(new Set(conversationIds));
};

export const cleanupOrphanedConversationStorage = (
  existingConversationIds: string[],
): void => {
  const storedConversationIds = findStoredConversationIds();
  const existingIds = new Set(existingConversationIds);

  storedConversationIds.forEach((conversationId) => {
    if (!existingIds.has(conversationId)) {
      clearConversationStorage(conversationId);
    }
  });
};
