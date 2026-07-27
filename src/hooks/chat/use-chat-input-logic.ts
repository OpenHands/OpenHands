import { useRef, useCallback, useEffect, useState } from "react";
import {
  isContentEmpty,
  clearEmptyContent,
  getTextContent,
} from "#/components/features/chat/utils/chat-input.utils";
import {
  useConversationStore,
  type IMessageToSend,
} from "#/stores/conversation-store";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useDraftPersistence } from "./use-draft-persistence";

/**
 * How recent a `messageToSend` must be for the home composer to honor it as an
 * intentional one-shot prefill (e.g. "Create Automation" / skill launch flows,
 * which navigate to the home composer and then set the message in a 0ms
 * timeout). Anything older is a stale leftover from a previous conversation
 * and must not clobber the just-restored sessionStorage draft.
 */
const HOME_PREFILL_MAX_AGE_MS = 5_000;

/**
 * Hook for managing chat input content logic
 */
export const useChatInputLogic = () => {
  const chatInputRef = useRef<HTMLDivElement | null>(null);
  // Optional because the chat input also renders on the home page, where no
  // conversation route is mounted yet. Draft persistence is conversation-
  // scoped, so it no-ops when this is undefined.
  const { conversationId } = useOptionalConversationId();

  const {
    messageToSend: rawMessageToSend,
    messageRestoreIfEmpty,
    hasRightPanelToggled,
    setMessageToSend,
    clearMessageRestoreIfEmpty,
    setIsRightPanelShown,
  } = useConversationStore();

  // Draft persistence - saves to localStorage/sessionStorage, restores on mount
  const { saveDraft, clearDraft } = useDraftPersistence(
    conversationId,
    chatInputRef,
  );

  // On the home page (no conversationId) a stale messageToSend value in the
  // Zustand store would cause useAutoResize to overwrite the just-restored
  // sessionStorage draft (see useAutoResize value effect), so stale values are
  // dropped here (null keeps value=undefined in useAutoResize so it never
  // touches the element content). Fresh values, however, are intentional
  // one-shot prefills: launch flows such as the "Create Automation" modal
  // navigate("/conversations") and then setMessageToSend just afterwards, and
  // the home composer must honor them (useAutoResize applies the text, places
  // the caret at the end, and clears the store value via onValueApplied).
  // The store timestamp distinguishes the two cases. The check runs in an
  // effect (not during render) because reading the clock in render is impure.
  const [homePrefill, setHomePrefill] = useState<IMessageToSend | null>(null);

  useEffect(() => {
    if (conversationId) {
      return;
    }
    if (!rawMessageToSend) {
      setHomePrefill(null);
      return;
    }
    if (Date.now() - rawMessageToSend.timestamp <= HOME_PREFILL_MAX_AGE_MS) {
      setHomePrefill(rawMessageToSend);
    }
  }, [conversationId, rawMessageToSend]);

  const messageToSend = conversationId ? rawMessageToSend : homePrefill;

  // Restore a cancelled pending send back into the input only when empty.
  useEffect(() => {
    if (!conversationId || !messageRestoreIfEmpty) {
      return;
    }

    const currentText = getTextContent(chatInputRef.current).trim();
    if (currentText.length === 0) {
      setMessageToSend(messageRestoreIfEmpty.text);
    }
    clearMessageRestoreIfEmpty();
  }, [
    conversationId,
    messageRestoreIfEmpty,
    setMessageToSend,
    clearMessageRestoreIfEmpty,
  ]);

  // Save current input value when drawer state changes (conversation view only)
  useEffect(() => {
    if (!conversationId) return;
    if (chatInputRef.current) {
      const currentText = getTextContent(chatInputRef.current);
      setMessageToSend(currentText);
      setIsRightPanelShown(hasRightPanelToggled);
    }
  }, [
    conversationId,
    hasRightPanelToggled,
    setMessageToSend,
    setIsRightPanelShown,
  ]);

  // Helper function to check if contentEditable is truly empty
  const checkIsContentEmpty = useCallback(
    (): boolean => isContentEmpty(chatInputRef.current),
    [],
  );

  // Helper function to properly clear contentEditable for placeholder display
  const clearEmptyContentHandler = useCallback((): void => {
    clearEmptyContent(chatInputRef.current);
  }, []);

  // Get current message text
  const getCurrentMessage = useCallback(
    (): string => getTextContent(chatInputRef.current),
    [],
  );

  return {
    chatInputRef,
    messageToSend,
    checkIsContentEmpty,
    clearEmptyContentHandler,
    getCurrentMessage,
    saveDraft,
    clearDraft,
  };
};
