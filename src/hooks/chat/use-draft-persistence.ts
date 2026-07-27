import { useEffect, useRef, useCallback, useState } from "react";
import {
  useConversationLocalStorageState,
  getConversationState,
  setConversationState,
} from "#/utils/conversation-local-storage";
import {
  focusContentEditableAtEnd,
  getTextContent,
} from "#/components/features/chat/utils/chat-input.utils";

/**
 * Check if a conversation ID is a temporary task ID.
 * Task IDs have the format "task-{uuid}" and are used during V1 conversation initialization.
 */
const isTaskId = (id: string): boolean => id.startsWith("task-");

const DRAFT_SAVE_DEBOUNCE_MS = 500;

/**
 * sessionStorage key used to persist the home-page prompt draft.
 * Cleared on successful conversation creation; survives navigation within the session.
 */
export const HOME_PROMPT_DRAFT_KEY = "oh:home-prompt-draft";

/**
 * Hook for persisting draft messages.
 * Handles debounced saving on input, restoration on mount, and clearing on confirmed delivery.
 *
 * When `conversationId` is defined, the draft is persisted to localStorage
 * under the conversation's key. When `conversationId` is undefined (home page),
 * the draft is persisted to sessionStorage under `HOME_PROMPT_DRAFT_KEY` so it
 * survives navigation within the session but is discarded on tab close.
 */
export const useDraftPersistence = (
  conversationId: string | null | undefined,
  chatInputRef: React.RefObject<HTMLDivElement | null>,
) => {
  // Pass an empty string when conversationId is missing to keep the hook
  // call unconditional (rules of hooks). The early-return branches below
  // ensure we never actually touch localStorage with an empty key.
  const { state, setDraftMessage } = useConversationLocalStorageState(
    conversationId ?? "",
  );
  const saveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hasRestoredRef = useRef(false);
  const [isRestored, setIsRestored] = useState(false);

  // Track current conversationId to prevent saving draft to wrong conversation
  const currentConversationIdRef = useRef(conversationId);
  // Track if this is the first mount to handle initial cleanup
  const isFirstMountRef = useRef(true);
  // Tracks the latest home-page text so the unmount flush can use it safely.
  // chatInputRef.current is null by the time async useEffect cleanup runs in
  // React 18 (refs are cleared during the synchronous commit phase, before
  // passive effects fire), so we can't read from the DOM there.
  const lastHomeTextRef = useRef<string>("");

  // IMPORTANT: This effect must run FIRST when conversation changes.
  // It handles three concerns:
  // 1. Cleanup: Cancel pending saves from previous conversation
  // 2. Task-to-real transition: Preserve draft typed during initialization
  // 3. DOM reset: Clear stale content before restoration effect runs
  useEffect(() => {
    if (!conversationId) {
      return;
    }
    const previousConversationId = currentConversationIdRef.current;
    const isInitialMount = isFirstMountRef.current;
    currentConversationIdRef.current = conversationId;
    isFirstMountRef.current = false;

    // --- 1. Cancel pending saves from previous conversation ---
    // Prevents draft from being saved to wrong conversation if user switched quickly
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
      saveTimeoutRef.current = null;
    }

    const element = chatInputRef.current;

    // --- 2. Handle task-to-real ID transition (preserve draft during initialization) ---
    // When a new V1 conversation initializes, it starts with a temporary "task-xxx" ID
    // that transitions to a real conversation ID once ready. Task IDs don't persist
    // to localStorage, so any draft typed during this phase would be lost.
    // We detect this transition and transfer the draft to the new real ID.
    if (
      !isInitialMount &&
      previousConversationId &&
      previousConversationId !== conversationId
    ) {
      const wasTaskId = isTaskId(previousConversationId);
      const isNowRealId = !isTaskId(conversationId);

      if (wasTaskId && isNowRealId && element) {
        const currentText = getTextContent(element).trim();
        if (currentText) {
          // Transfer draft to the new (real) conversation ID
          setConversationState(conversationId, { draftMessage: currentText });
          // Keep draft visible in DOM and mark as restored to prevent overwrite
          hasRestoredRef.current = true;
          setIsRestored(true);
          return; // Skip normal cleanup - draft is already in correct state
        }
      }
    }

    // --- 3. Clear stale DOM content (will be restored by next effect if draft exists) ---
    // This prevents stale drafts from appearing in new conversations due to:
    // - Browser form restoration on back/forward navigation
    // - React DOM recycling between conversation switches
    // The restoration effect will then populate with the correct saved draft
    if (element) {
      element.textContent = "";
    }

    // Reset restoration flag so the restoration effect will run for new conversation
    hasRestoredRef.current = false;
    setIsRestored(false);
  }, [conversationId, chatInputRef]);

  // Restore draft on mount - uses sessionStorage for the home page (no conversationId)
  // and localStorage for active conversations.
  useEffect(() => {
    if (hasRestoredRef.current) {
      return;
    }

    const element = chatInputRef.current;
    if (!element) {
      return;
    }

    if (!conversationId) {
      // Home page: restore from sessionStorage
      try {
        const draft = sessionStorage.getItem(HOME_PROMPT_DRAFT_KEY);
        if (draft && getTextContent(element).trim() === "") {
          element.textContent = draft;
          const selection = window.getSelection();
          const range = document.createRange();
          range.selectNodeContents(element);
          range.collapse(false);
          selection?.removeAllRanges();
          selection?.addRange(range);
          // Seed lastHomeTextRef so an unmount flush without any typing still
          // preserves the restored text rather than clearing sessionStorage.
          lastHomeTextRef.current = draft;
        }
      } catch {
        // sessionStorage not available
      }
      hasRestoredRef.current = true;
      setIsRestored(true);
      return;
    }

    // Read directly from localStorage to avoid stale state from useConversationLocalStorageState
    // The hook's state may not have synced yet after conversationId change
    const { draftMessage } = getConversationState(conversationId);

    // Only restore if there's a saved draft and the input is empty
    if (draftMessage && getTextContent(element).trim() === "") {
      element.textContent = draftMessage;
      focusContentEditableAtEnd(element);
    }

    hasRestoredRef.current = true;
    setIsRestored(true);
  }, [chatInputRef, conversationId]);

  // Debounced save function - called from onInput handler
  const saveDraft = useCallback(() => {
    // Clear any pending save
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }

    if (!conversationId) {
      // Home page: write to sessionStorage synchronously so there is no
      // debounce window in which navigation can discard the latest text.
      const element = chatInputRef.current;
      if (element) {
        const text = getTextContent(element).trim();
        lastHomeTextRef.current = text;
        try {
          if (text) {
            sessionStorage.setItem(HOME_PROMPT_DRAFT_KEY, text);
          } else {
            sessionStorage.removeItem(HOME_PROMPT_DRAFT_KEY);
          }
        } catch {
          // sessionStorage not available
        }
      }
      return;
    }

    // Capture the conversationId at the time of input
    const capturedConversationId = conversationId;

    saveTimeoutRef.current = setTimeout(() => {
      // Verify we're still on the same conversation before saving
      // This prevents saving draft to wrong conversation if user switched quickly
      if (capturedConversationId !== currentConversationIdRef.current) {
        return;
      }

      const element = chatInputRef.current;
      if (!element) {
        return;
      }

      const text = getTextContent(element).trim();
      // Only save if content has changed
      if (text !== (state.draftMessage || "")) {
        setDraftMessage(text || null);
      }
    }, DRAFT_SAVE_DEBOUNCE_MS);
  }, [chatInputRef, state.draftMessage, setDraftMessage, conversationId]);

  // Clear draft - called after message delivery is confirmed
  const clearDraft = useCallback(() => {
    // Cancel any pending save
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
      saveTimeoutRef.current = null;
    }
    if (!conversationId) {
      // Home page: clear sessionStorage
      try {
        sessionStorage.removeItem(HOME_PROMPT_DRAFT_KEY);
      } catch {
        // sessionStorage not available
      }
      return;
    }
    setDraftMessage(null);
  }, [conversationId, setDraftMessage]);

  // Cleanup on unmount: cancel any pending debounce timer and, for the home
  // page, flush the last-tracked text to sessionStorage. We read from
  // lastHomeTextRef rather than chatInputRef because React clears ref.current
  // during the synchronous commit phase — before async useEffect cleanups run
  // — so the DOM ref is null by the time this function executes.
  //
  // We only write text back if the key already exists in sessionStorage.
  // saveDraft writes synchronously on every keystroke, so the key is present
  // whenever there is unsaved text. If the key is absent it was intentionally
  // removed — most importantly by HomeChatLauncher.onSuccess after a
  // successful conversation start — and we must not restore it here.
  useEffect(
    () => () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
      if (!currentConversationIdRef.current) {
        const text = lastHomeTextRef.current;
        try {
          if (text) {
            if (sessionStorage.getItem(HOME_PROMPT_DRAFT_KEY) !== null) {
              sessionStorage.setItem(HOME_PROMPT_DRAFT_KEY, text);
            }
          } else {
            sessionStorage.removeItem(HOME_PROMPT_DRAFT_KEY);
          }
        } catch {
          // sessionStorage not available
        }
      }
    },
    [],
  );

  return {
    saveDraft,
    clearDraft,
    isRestored,
    hasDraft: !!state.draftMessage,
  };
};
