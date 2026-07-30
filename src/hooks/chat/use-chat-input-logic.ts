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

  // A one-shot prefill belongs to exactly one composer. Explicit targeting
  // avoids timing heuristics and prevents a value queued for home (or another
  // conversation) from overwriting this composer's restored draft.
  const messageToSend =
    rawMessageToSend?.targetConversationId === (conversationId ?? null)
      ? rawMessageToSend
      : null;

  // Restore a cancelled pending send back into the input only when empty.
  useEffect(() => {
    if (
      !conversationId ||
      messageRestoreIfEmpty?.targetConversationId !== conversationId
    ) {
      return;
    }

    const currentText = getTextContent(chatInputRef.current).trim();
    if (currentText.length === 0) {
      setMessageToSend(messageRestoreIfEmpty.text, conversationId);
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
      setMessageToSend(currentText, conversationId);
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
