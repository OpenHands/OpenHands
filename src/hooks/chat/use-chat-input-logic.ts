import { useRef, useCallback, useEffect } from "react";
import {
  isContentEmpty,
  clearEmptyContent,
  getTextContent,
} from "#/components/features/chat/utils/chat-input.utils";
import { useConversationStore } from "#/stores/conversation-store";
import { useNavigation } from "#/context/navigation-context";
import { useDraftPersistence } from "./use-draft-persistence";

/**
 * Hook for managing chat input content logic
 */
export const useChatInputLogic = () => {
  const chatInputRef = useRef<HTMLDivElement | null>(null);
  // The chat input also renders on the home page, where no conversation route
  // is mounted yet. Draft persistence is conversation-scoped, so it no-ops
  // when conversationId is undefined.
  const { conversationId, isNavigating } = useNavigation();

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
  // mechanism is not relevant.  More importantly, a stale *empty* messageToSend
  // value in the Zustand store causes useAutoResize to overwrite the
  // just-restored sessionStorage draft with an empty string (see useAutoResize
  // value effect). Empty values are still filtered to null so useAutoResize
  // keeps value=undefined and never touches the element content on the home
  // page. Non-empty seeded prompts (e.g. the automation "Create Automation"
  // flow) must pass through, otherwise the home-page input renders blank.
  // While a navigation is in flight this page can still be mounted even though
  // a prompt was seeded for the destination route (producers inject via
  // setTimeout right after navigate), so home must not expose — and one-shot
  // consume — a value addressed to the next page's composer.
  const messageToSend =
    conversationId ||
    (!isNavigating && (rawMessageToSend?.text.trim().length ?? 0) > 0)
      ? rawMessageToSend
      : null;

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
