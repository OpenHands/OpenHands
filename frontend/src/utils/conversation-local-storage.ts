import { useState } from "react";
import type { ConversationTab } from "#/stores/conversation-store";

export const LOCAL_STORAGE_KEYS = {
  CONVERSATION_STATE: "conversation-state",
} as const;

/**
 * Consolidated conversation state stored in a single localStorage key.
 */
export interface ConversationState {
  selectedTab: ConversationTab | null;
  rightPanelShown: boolean;
  unpinnedTabs: string[];
}

const DEFAULT_CONVERSATION_STATE: ConversationState = {
  selectedTab: "editor",
  rightPanelShown: true,
  unpinnedTabs: [],
};

/**
 * Check if a conversation ID is a temporary task ID that should not be persisted.
 * Task IDs have the format "task-{uuid}" and are used during V1 conversation initialization.
 */
export function isTaskConversationId(conversationId: string): boolean {
  return conversationId.startsWith("task-");
}

/**
 * Get the full conversation state from localStorage.
 */
export function getConversationState(
  conversationId: string,
): ConversationState {
  if (isTaskConversationId(conversationId)) {
    return DEFAULT_CONVERSATION_STATE;
  }
  try {
    const key = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;
    const item = localStorage.getItem(key);
    if (item !== null) {
      return { ...DEFAULT_CONVERSATION_STATE, ...JSON.parse(item) };
    }
    return DEFAULT_CONVERSATION_STATE;
  } catch {
    return DEFAULT_CONVERSATION_STATE;
  }
}

/**
 * Set the conversation state in localStorage, merging with existing state.
 */
export function setConversationState(
  conversationId: string,
  updates: Partial<ConversationState>,
): void {
  if (isTaskConversationId(conversationId)) {
    return;
  }
  try {
    const key = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;
    const currentState = getConversationState(conversationId);
    const newState = { ...currentState, ...updates };
    localStorage.setItem(key, JSON.stringify(newState));
  } catch (err) {
    console.warn("Failed to set conversation localStorage", err);
  }
}

export function clearConversationLocalStorage(conversationId: string) {
  try {
    const key = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;
    localStorage.removeItem(key);
  } catch (err) {
    console.warn(
      "Failed to clear conversation localStorage",
      conversationId,
      err,
    );
  }
}

/**
 * React hook for conversation-scoped localStorage state.
 * Returns the full state and individual setters for each property.
 */
export function useConversationLocalStorageState(conversationId: string): {
  state: ConversationState;
  setSelectedTab: (tab: ConversationTab | null) => void;
  setRightPanelShown: (shown: boolean) => void;
  setUnpinnedTabs: (tabs: string[]) => void;
} {
  const [state, setState] = useState<ConversationState>(() =>
    getConversationState(conversationId),
  );

  const updateState = (updates: Partial<ConversationState>) => {
    setState((prev) => ({ ...prev, ...updates }));
    setConversationState(conversationId, updates);
  };

  return {
    state,
    setSelectedTab: (tab) => updateState({ selectedTab: tab }),
    setRightPanelShown: (shown) => updateState({ rightPanelShown: shown }),
    setUnpinnedTabs: (tabs) => updateState({ unpinnedTabs: tabs }),
  };
}
