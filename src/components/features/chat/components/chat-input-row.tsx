import React from "react";
import { ChatInputField } from "./chat-input-field";

interface ChatInputRowProps {
  chatInputRef: React.RefObject<HTMLDivElement | null>;
  isNewConversationPending?: boolean;
  onInput: () => void;
  onPaste: (e: React.ClipboardEvent) => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
  onFocus?: () => void;
  onBlur?: () => void;
}

export function ChatInputRow({
  chatInputRef,
  isNewConversationPending = false,
  onInput,
  onPaste,
  onKeyDown,
  onFocus,
  onBlur,
}: ChatInputRowProps) {
  return (
    <div className="box-border content-stretch flex flex-row items-end justify-between p-0 relative shrink-0 w-full pb-[18px] gap-2">
      <div className="basis-0 box-border content-stretch flex flex-row gap-4 grow items-end justify-start min-h-px min-w-px p-0 relative shrink-0">
        <ChatInputField
          chatInputRef={chatInputRef}
          disabled={isNewConversationPending}
          onInput={onInput}
          onPaste={onPaste}
          onKeyDown={onKeyDown}
          onFocus={onFocus}
          onBlur={onBlur}
        />
      </div>
    </div>
  );
}
