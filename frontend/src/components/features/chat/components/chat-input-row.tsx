import React from "react";
import { cn } from "#/utils/utils";
import { ChatAddFileButton } from "../chat-add-file-button";
import { ChatSendButton } from "../chat-send-button";
import { ChatInputField } from "./chat-input-field";

interface ChatInputRowProps {
  chatInputRef: React.RefObject<HTMLDivElement | null>;
  disabled: boolean;
  isNewConversationPending?: boolean;
  showButton: boolean;
  buttonClassName: string;
  handleFileIconClick: (isDisabled: boolean) => void;
  handleSubmit: () => void;
  onInput: () => void;
  onPaste: (e: React.ClipboardEvent) => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
  onFocus?: () => void;
  onBlur?: () => void;
}

export function ChatInputRow({
  chatInputRef,
  disabled,
  isNewConversationPending = false,
  showButton,
  buttonClassName,
  handleFileIconClick,
  handleSubmit,
  onInput,
  onPaste,
  onKeyDown,
  onFocus,
  onBlur,
}: ChatInputRowProps) {
  return (
    <div className="box-border content-stretch flex flex-row items-end justify-between p-0 relative shrink-0 w-full pb-[18px] gap-2">
      <div className="basis-0 box-border content-stretch flex flex-row gap-4 grow items-end justify-start min-h-px min-w-px p-0 relative shrink-0">
        <ChatAddFileButton
          disabled={disabled}
          handleFileIconClick={() => handleFileIconClick(disabled)}
        />

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

      {/* Send Button */}
      {showButton && (
        <ChatSendButton
          buttonClassName={cn(buttonClassName, "translate-y-[3px]")}
          handleSubmit={handleSubmit}
          disabled={disabled}
        />
      )}
    </div>
  );
}
