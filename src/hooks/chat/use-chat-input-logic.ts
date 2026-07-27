import { useRef, useCallback, useEffect } from "react";
import {
  isContentEmpty,
  clearEmptyContent,
  getTextContent,
} from "#/components/features/chat/utils/chat-input.utils";
import { useConversationStore } from "#/stores/conversation-store";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useDraftPersistence } from "./use-draft-persistence";

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

  // On the home page (no conversationId) the right-panel / messageToSend
  // mechanism is not relevant.  More importantly, a stale messageToSend value
  // in the Zustand store causes useAutoResize to overwrite the just-restored
  // sessionStorage draft with an empty string (see useAutoResize value effect).
  // Returning null here keeps value=undefined in useAutoResize so it never
  // touches the element content on the home page.
  const messageToSend = conversationId ? rawMessageToSend : null;

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
