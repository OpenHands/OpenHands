import React from "react";
import { DragOver } from "../drag-over";
import { UploadedFiles } from "../uploaded-files";
import { ChatInputRow } from "./chat-input-row";
import { ChatInputActions } from "./chat-input-actions";
import { useConversationStore } from "#/stores/conversation-store";
import { cn } from "#/utils/utils";

interface ChatInputContainerProps {
  chatContainerRef: React.RefObject<HTMLDivElement | null>;
  isDragOver: boolean;
  disabled: boolean;
  showButton: boolean;
  buttonClassName: string;
  chatInputRef: React.RefObject<HTMLDivElement | null>;
  handleFileIconClick: (isDisabled: boolean) => void;
  handleSubmit: () => void;
  handleResumeAgent: () => void;
  onDragOver: (e: React.DragEvent, isDisabled: boolean) => void;
  onDragLeave: (e: React.DragEvent, isDisabled: boolean) => void;
  onDrop: (e: React.DragEvent, isDisabled: boolean) => void;
  onInput: () => void;
  onPaste: (e: React.ClipboardEvent) => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
  onFocus?: () => void;
  onBlur?: () => void;
}

export function ChatInputContainer({
  chatContainerRef,
  isDragOver,
  disabled,
  showButton,
  buttonClassName,
  chatInputRef,
  handleFileIconClick,
  handleSubmit,
  handleResumeAgent,
  onDragOver,
  onDragLeave,
  onDrop,
  onInput,
  onPaste,
  onKeyDown,
  onFocus,
  onBlur,
}: ChatInputContainerProps) {
  const conversationMode = useConversationStore(
    (state) => state.conversationMode,
  );

  return (
    <div
      ref={chatContainerRef}
      className={cn(
        "bg-[#25272D] box-border content-stretch flex flex-col items-start justify-center p-4 pt-3 relative rounded-[15px] w-full",
        conversationMode === "plan" && "border border-[#597FF4]",
      )}
      onDragOver={(e) => onDragOver(e, disabled)}
      onDragLeave={(e) => onDragLeave(e, disabled)}
      onDrop={(e) => onDrop(e, disabled)}
    >
      {/* Drag Over UI */}
      {isDragOver && <DragOver />}

      <UploadedFiles />

      <ChatInputRow
        chatInputRef={chatInputRef}
        disabled={disabled}
        showButton={showButton}
        buttonClassName={buttonClassName}
        handleFileIconClick={handleFileIconClick}
        handleSubmit={handleSubmit}
        onInput={onInput}
        onPaste={onPaste}
        onKeyDown={onKeyDown}
        onFocus={onFocus}
        onBlur={onBlur}
      />

      <ChatInputActions
        disabled={disabled}
        handleResumeAgent={handleResumeAgent}
      />
    </div>
  );
}
