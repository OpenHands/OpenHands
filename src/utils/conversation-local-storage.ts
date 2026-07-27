import { useEffect, useState } from "react";
import type {
  ConversationTab,
  ConversationMode,
} from "#/stores/conversation-store";
import type { ViewMode } from "#/components/features/files-tab/view-mode";

export const LOCAL_STORAGE_KEYS = {
  CONVERSATION_STATE: "conversation-state",
  PENDING_TASK_DRAFT: "pending-task-draft",
} as const;

const CONVERSATION_STATE_UPDATED_EVENT = "conversation-state-updated";

type ConversationStateUpdatedDetail = {
  conversationId: string;
};

/**
 * Consolidated conversation state stored in a single localStorage key.
 *
 * NOTE: the right-drawer open/closed state is intentionally *not* persisted
 * here. Users expect the drawer to start closed every time the app is
 * (re)loaded and only stay open when they opened it themselves during the
 * current session. That state lives in the in-memory Zustand
 * `useConversationStore` (`isRightPanelShown` / `hasRightPanelToggled`)
 * which survives in-app navigation but resets on full reloads. Older
 * builds wrote a `rightPanelShown` field into this blob; we accept and
 * silently ignore that field on read so we don't churn old localStorage.
 */
export interface ConversationState {
  selectedTab: ConversationTab | null;
  unpinnedTabs: string[];
  conversationMode: ConversationMode;
  subConversationTaskId: string | null;
  draftMessage: string | null;
  /**
   * User's persisted choice for the Files tab diff-vs-files toggle.
   * `null` means "no explicit choice yet" — the Files tab then falls back
   * to its repo-aware default (diff inside a git repo with commits, files
   * otherwise).
   */
  filesTabDiffView: boolean | null;
  /** User's persisted choice for the Files tab Rich/Plain content toggle. */
  filesTabContentViewMode: ViewMode;
}

const DEFAULT_CONVERSATION_STATE: ConversationState = {
  selectedTab: "files",
  unpinnedTabs: [],
  conversationMode: "code",
  subConversationTaskId: null,
  draftMessage: null,
  filesTabDiffView: null,
  filesTabContentViewMode: "rich",
};

const VALID_CONVERSATION_TABS: ReadonlySet<ConversationTab> = new Set([
  "files",
  "browser",
  "terminal",
  "planner",
  "tasklist",
]);

// Tab keys that *used to* exist and were removed during the Files tab
// refactor. We strip these out of any persisted state on read so that
// returning users don't keep ghost entries (in `selectedTab` or
// `unpinnedTabs`) forever — the UI has no way to surface them again to
// be re-pinned or re-selected. We deliberately use an explicit denylist
// here rather than a strict "whitelist against VALID_CONVERSATION_TABS"
// filter because `unpinnedTabs` is typed `string[]` and other parts of
// the app legitimately store arbitrary tab keys in it (sub-conversation
// tab ids etc.).
const REMOVED_CONVERSATION_TABS: ReadonlySet<string> = new Set([
  "editor",
  "served",
  "changes",
  "app",
  "vscode",
]);

const VALID_VIEW_MODES: ReadonlySet<ViewMode> = new Set(["rich", "plain"]);

function sanitizeStoredState(
  stored: Record<string, unknown>,
): Partial<ConversationState> {
  // `rightPanelShown` is no longer part of the schema (the drawer's
  // open state is session-only) but old persisted blobs still carry it.
  // We just drop it on the way in — keeping it in `result` here would
  // pollute the merged `ConversationState` shape that consumers receive.
  const { rightPanelShown: _ignoredRightPanelShown, ...rest } = stored as {
    rightPanelShown?: unknown;
  } & Record<string, unknown>;
  let result: Partial<ConversationState> = rest as Partial<ConversationState>;

  if (
    result.selectedTab != null &&
    (REMOVED_CONVERSATION_TABS.has(result.selectedTab) ||
      !VALID_CONVERSATION_TABS.has(result.selectedTab as ConversationTab))
  ) {
    result = { ...result };
    delete result.selectedTab;
  }

  if (result.unpinnedTabs) {
    const filtered = result.unpinnedTabs.filter(
      (tab) => !REMOVED_CONVERSATION_TABS.has(tab),
    );
    if (filtered.length !== result.unpinnedTabs.length) {
      result = { ...result, unpinnedTabs: filtered };
    }
  }

  // Drop a corrupt/manually-edited `filesTabContentViewMode` so it falls
  // back to the typed default ("rich") instead of leaking a junk string
  // into ViewMode-typed consumers. TypeScript can't help us here — the
  // value comes from `JSON.parse(localStorage.getItem(...))` which is
  // unknown at runtime.
  if (
    result.filesTabContentViewMode != null &&
    !VALID_VIEW_MODES.has(result.filesTabContentViewMode)
  ) {
    result = { ...result };
    delete result.filesTabContentViewMode;
  }

  return result;
}

/**
 * Check if a conversation ID is a temporary task ID that should not be persisted.
 * Task IDs have the format "task-{uuid}" and are used during V1 conversation initialization.
 */
export function isTaskConversationId(conversationId: string): boolean {
  return conversationId.startsWith("task-");
}

/**
 * Whether persistence should be skipped for this conversation id.
 *
 * Skips:
 *  - empty string ids (callers outside of a conversation route, e.g.
 *    rendered inside a unit test without a NavigationProvider)
 *  - "task-..." ids used as placeholders during V1 conversation
 *    initialization
 */
function shouldSkipPersistence(conversationId: string): boolean {
  return conversationId === "" || isTaskConversationId(conversationId);
}

/**
 * Get the full conversation state from localStorage.
 */
export function getConversationState(
  conversationId: string,
): ConversationState {
  if (shouldSkipPersistence(conversationId)) {
    return DEFAULT_CONVERSATION_STATE;
  }
  try {
    const key = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;
    const item = localStorage.getItem(key);
    if (item !== null) {
      return {
        ...DEFAULT_CONVERSATION_STATE,
        ...sanitizeStoredState(JSON.parse(item)),
      };
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
  if (shouldSkipPersistence(conversationId)) {
    return;
  }
  try {
    const key = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;
    const currentState = getConversationState(conversationId);
    const newState = { ...currentState, ...updates };
    localStorage.setItem(key, JSON.stringify(newState));
    if (typeof window !== "undefined") {
      window.dispatchEvent(
        new CustomEvent<ConversationStateUpdatedDetail>(
          CONVERSATION_STATE_UPDATED_EVENT,
          { detail: { conversationId } },
        ),
      );
    }
  } catch (err) {
    console.warn("Failed to set conversation localStorage", err);
  }
}

function getPendingTaskDraftKey(taskId: string): string {
  return `${LOCAL_STORAGE_KEYS.PENDING_TASK_DRAFT}-${taskId}`;
}

export function setPendingTaskDraft(
  taskId: string | null | undefined,
  draftMessage: string,
): void {
  if (!taskId) return;
  try {
    localStorage.setItem(getPendingTaskDraftKey(taskId), draftMessage);
  } catch (err) {
    console.warn("Failed to store pending task draft", err);
  }
}

export function consumePendingTaskDraft(
  taskId: string | null | undefined,
): string | null {
  if (!taskId) return null;
  try {
    const key = getPendingTaskDraftKey(taskId);
    const draft = localStorage.getItem(key);
    localStorage.removeItem(key);
    return draft;
  } catch (err) {
    console.warn("Failed to consume pending task draft", err);
    return null;
  }
}

export function clearConversationLocalStorage(conversationId: string) {
  try {
    const key = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;
    localStorage.removeItem(key);
    if (typeof window !== "undefined") {
      window.dispatchEvent(
        new CustomEvent<ConversationStateUpdatedDetail>(
          CONVERSATION_STATE_UPDATED_EVENT,
          { detail: { conversationId } },
        ),
      );
    }
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
 *
 * The right-drawer open state is deliberately not represented here —
 * see the note on `ConversationState` for the rationale.
 */
export function useConversationLocalStorageState(conversationId: string): {
  state: ConversationState;
  setSelectedTab: (tab: ConversationTab | null) => void;
  setUnpinnedTabs: (tabs: string[]) => void;
  setConversationMode: (mode: ConversationMode) => void;
  setDraftMessage: (message: string | null) => void;
  setFilesTabDiffView: (diffView: boolean | null) => void;
  setFilesTabContentViewMode: (mode: ViewMode) => void;
} {
  const [state, setState] = useState<ConversationState>(() =>
    getConversationState(conversationId),
  );

  useEffect(() => {
    if (typeof window === "undefined") {
      return undefined;
    }

    const key = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;

    const syncState = () => {
      setState(getConversationState(conversationId));
    };

    const handleStorage = (event: StorageEvent) => {
      if (event.key === key) {
        syncState();
      }
    };

    const handleConversationStateUpdated = (event: Event) => {
      const customEvent = event as CustomEvent<ConversationStateUpdatedDetail>;
      if (customEvent.detail?.conversationId === conversationId) {
        syncState();
      }
    };

    // Ensure this hook reflects latest state for the current conversation ID.
    syncState();

    window.addEventListener("storage", handleStorage);
    window.addEventListener(
      CONVERSATION_STATE_UPDATED_EVENT,
      handleConversationStateUpdated,
    );

    return () => {
      window.removeEventListener("storage", handleStorage);
      window.removeEventListener(
        CONVERSATION_STATE_UPDATED_EVENT,
        handleConversationStateUpdated,
      );
    };
  }, [conversationId]);

  const updateState = (updates: Partial<ConversationState>) => {
    if (shouldSkipPersistence(conversationId)) {
      // No durable storage for this id (empty / task placeholder), but the
      // hook is still useful as ephemeral in-memory state — update the
      // local React mirror directly so toggles in the UI behave normally
      // until a real conversation id arrives.
      setState((prev) => ({ ...prev, ...updates }));
      return;
    }
    setConversationState(conversationId, updates);
  };

  return {
    state,
    setSelectedTab: (tab) => updateState({ selectedTab: tab }),
    setUnpinnedTabs: (tabs) => updateState({ unpinnedTabs: tabs }),
    setConversationMode: (mode) => updateState({ conversationMode: mode }),
    setDraftMessage: (message) => updateState({ draftMessage: message }),
    setFilesTabDiffView: (diffView) =>
      updateState({ filesTabDiffView: diffView }),
    setFilesTabContentViewMode: (mode) =>
      updateState({ filesTabContentViewMode: mode }),
  };
}
