import React, { useEffect } from "react";
import { DragOver } from "../drag-over";
import { QuoteCard } from "../quote-card";
import { UploadedFiles } from "../uploaded-files";
import { ChatInputRow } from "./chat-input-row";
import { ChatInputActions } from "./chat-input-actions";
import { SlashCommandMenu } from "./slash-command-menu";
import { useConversationStore } from "#/stores/conversation-store";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import {
  usePendingReplyQuote,
  useReplyQuoteStore,
} from "#/stores/reply-quote-store";
import { cn } from "#/utils/utils";
import { SlashCommandItem } from "#/hooks/chat/use-slash-command";
import { useJumpToEvent } from "#/hooks/chat/use-jump-to-event";
import { insertTextAtCaret } from "../utils/chat-input.utils";

interface ChatInputContainerProps {
  chatContainerRef: React.RefObject<HTMLDivElement | null>;
  isDragOver: boolean;
  disabled: boolean;
  canSubmit: boolean;
  hasStartedConversation?: boolean;
  isNewConversationPending?: boolean;
  showButton: boolean;
  buttonClassName: string;
  chatInputRef: React.RefObject<HTMLDivElement | null>;
  placeholder?: string;
  handleFileIconClick: (isDisabled: boolean) => void;
  handleSubmit: () => void;
  onDragOver: (e: React.DragEvent, isDisabled: boolean) => void;
  onDragLeave: (e: React.DragEvent, isDisabled: boolean) => void;
  onDrop: (e: React.DragEvent, isDisabled: boolean) => void;
  onInput: () => void;
  onPaste: (e: React.ClipboardEvent) => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
  onFocus?: () => void;
  onBlur?: () => void;
  isSlashMenuOpen?: boolean;
  slashItems?: SlashCommandItem[];
  slashSelectedIndex?: number;
  onSlashSelect?: (item: SlashCommandItem) => void;
}

export function ChatInputContainer({
  chatContainerRef,
  isDragOver,
  disabled,
  canSubmit,
  hasStartedConversation,
  isNewConversationPending = false,
  showButton,
  buttonClassName,
  chatInputRef,
  placeholder,
  handleFileIconClick,
  handleSubmit,
  onDragOver,
  onDragLeave,
  onDrop,
  onInput,
  onPaste,
  onKeyDown,
  onFocus,
  onBlur,
  isSlashMenuOpen = false,
  slashItems = [],
  slashSelectedIndex = 0,
  onSlashSelect,
}: ChatInputContainerProps) {
  const conversationMode = useConversationStore(
    (state) => state.conversationMode,
  );

  return (
    <div
      ref={chatContainerRef}
      className={cn(
        "bg-surface box-border content-stretch flex flex-col items-start justify-center p-4 relative rounded-[15px] w-full",
        conversationMode === "plan" && "border border-[#597FF4]",
      )}
      onDragOver={(e) => onDragOver(e, disabled)}
      onDragLeave={(e) => onDragLeave(e, disabled)}
      onDrop={(e) => onDrop(e, disabled)}
    >
      {/* Drag Over UI */}
      {isDragOver && <DragOver />}

      <QuoteSlot />

      <UploadedFiles />

      {/* Wrapper so the slash menu anchors just above the input row,
          not above the entire (possibly resized) container */}
      <div className="relative w-full">
        {isSlashMenuOpen && onSlashSelect && (
          <SlashCommandMenu
            items={slashItems}
            selectedIndex={slashSelectedIndex}
            onSelect={onSlashSelect}
          />
        )}

        <ChatInputRow
          chatInputRef={chatInputRef}
          isNewConversationPending={isNewConversationPending}
          placeholder={placeholder}
          onInput={onInput}
          onPaste={onPaste}
          onKeyDown={onKeyDown}
          onFocus={onFocus}
          onBlur={onBlur}
        />
      </div>

      <ChatInputActions
        disabled={disabled}
        canSubmit={canSubmit}
        hasStartedConversation={hasStartedConversation}
        onAddFileClick={() => handleFileIconClick(disabled)}
        showButton={showButton}
        buttonClassName={buttonClassName}
        handleSubmit={handleSubmit}
        onDictationTranscript={(text) =>
          insertTextAtCaret(chatInputRef.current, text)
        }
        isDictationDisabled={isNewConversationPending}
      />
    </div>
  );
}

/**
 * The armed reply-quote card, rendered above the input row. Escape
 * dismisses it (keyboard parity with the card's X button).
 */
function QuoteSlot() {
  const { conversationId } = useOptionalConversationId();
  const quote = usePendingReplyQuote(conversationId ?? "");
  const jumpToEvent = useJumpToEvent();
  useEffect(() => {
    if (!quote) return undefined;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        useReplyQuoteStore.getState().clearReplyQuote(quote.conversationId);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [quote]);

  if (!conversationId || !quote) return null;

  return (
    <QuoteCard
      quote={quote}
      onJump={() => jumpToEvent(quote.sourceEventId)}
      onDismiss={() =>
        useReplyQuoteStore.getState().clearReplyQuote(quote.conversationId)
      }
    />
  );
}
