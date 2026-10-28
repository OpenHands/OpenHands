import { useCallback } from "react";
import {
  clearTextContent,
  clearFileInput,
} from "#/components/features/chat/utils/chat-input.utils";
import { composeReplyText, isBuiltInSlashCommand } from "#/utils/message-quote";
import { useConversationStore } from "#/stores/conversation-store";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import {
  usePendingReplyQuote,
  useReplyQuoteStore,
} from "#/stores/reply-quote-store";

/**
 * Hook for handling chat message submission
 */
export const useChatSubmission = (
  chatInputRef: React.RefObject<HTMLDivElement | null>,
  fileInputRef: React.RefObject<HTMLInputElement | null>,
  smartResize: () => void,
  onSubmit: (message: string) => void,
  resetManualResize?: () => void,
) => {
  const { conversationId } = useOptionalConversationId();
  const pendingReplyQuote = usePendingReplyQuote(conversationId ?? "");

  // Send button click handler
  const handleSubmit = useCallback(() => {
    const message = chatInputRef.current?.innerText || "";
    const trimmedMessage = message.trim();
    const { images, files } = useConversationStore.getState();
    const hasAttachments = images.length > 0 || files.length > 0;

    if (!trimmedMessage && !hasAttachments) {
      return;
    }

    // The armed quote composes into the outgoing text in the same
    // transaction that reads the composer, then disarms in the same
    // conversation the quote was armed for.
    // A built-in slash command is intercepted downstream and never sent as
    // text — composing an envelope around it would break the command match
    // and leak envelope syntax to the agent. Submit raw and keep the quote
    // armed for the user's next real message.
    const isSlashCommand = isBuiltInSlashCommand(message);
    onSubmit(
      isSlashCommand ? message : composeReplyText(pendingReplyQuote, message),
    );
    if (pendingReplyQuote && !isSlashCommand) {
      useReplyQuoteStore
        .getState()
        .clearReplyQuote(pendingReplyQuote.conversationId);
    }

    // Clear the input
    clearTextContent(chatInputRef.current);
    clearFileInput(fileInputRef.current);

    // Reset height and show suggestions again
    smartResize();

    // Reset manual resize state for next message
    resetManualResize?.();
  }, [
    chatInputRef,
    fileInputRef,
    smartResize,
    onSubmit,
    resetManualResize,
    pendingReplyQuote,
  ]);

  // Handle stop button click
  const handleStop = useCallback((onStop?: () => void) => {
    if (onStop) {
      onStop();
    }
  }, []);

  return {
    handleSubmit,
    handleStop,
  };
};
