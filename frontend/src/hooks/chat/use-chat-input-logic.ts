import { useRef, useCallback, useEffect } from "react";
import {
  isContentEmpty,
  clearEmptyContent,
  getTextContent,
} from "#/components/features/chat/utils/chat-input.utils";
import { useConversationStore } from "#/stores/conversation-store";

/**
 * Hook for managing chat input content logic
 */
export const useChatInputLogic = () => {
  const chatInputRef = useRef<HTMLDivElement | null>(null);

  const {
    messageToSend,
    hasRightPanelToggled,
    setMessageToSend,
    setIsRightPanelShown,
  } = useConversationStore();

  // Save current input value when drawer state changes
  useEffect(() => {
    if (chatInputRef.current) {
      const currentText = getTextContent(chatInputRef.current);
      setMessageToSend(currentText);
      setIsRightPanelShown(hasRightPanelToggled);
    }
  }, [hasRightPanelToggled, setMessageToSend, setIsRightPanelShown]);

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
  };
};
