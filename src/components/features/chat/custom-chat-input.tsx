import React, { useEffect, useRef } from "react";
import { useChatInputLogic } from "#/hooks/chat/use-chat-input-logic";
import { useFileHandling } from "#/hooks/chat/use-file-handling";
import { useGripResize } from "#/hooks/chat/use-grip-resize";
import { useChatInputEvents } from "#/hooks/chat/use-chat-input-events";
import { useChatSubmission } from "#/hooks/chat/use-chat-submission";
import { useSlashCommand } from "#/hooks/chat/use-slash-command";
import { ChatInputGrip } from "./components/chat-input-grip";
import { ChatInputContainer } from "./components/chat-input-container";
import { HiddenFileInput } from "./components/hidden-file-input";
import { useConversationStore } from "#/stores/conversation-store";
import { cn } from "#/utils/utils";

export interface CustomChatInputProps {
  disabled?: boolean;
  isNewConversationPending?: boolean;
  hasStartedConversation?: boolean;
  showButton?: boolean;
  onSubmit: (message: string) => void;
  onFocus?: () => void;
  onBlur?: () => void;
  onFilesPaste?: (
    files: File[],
    options?: import("#/hooks/chat/use-chat-attachment-upload").ChatAttachmentUploadOptions,
  ) => void;
  className?: React.HTMLAttributes<HTMLDivElement>["className"];
  buttonClassName?: React.HTMLAttributes<HTMLButtonElement>["className"];
}

export function CustomChatInput({
  disabled = false,
  isNewConversationPending = false,
  hasStartedConversation,
  showButton = true,
  onSubmit,
  onFocus,
  onBlur,
  onFilesPaste,
  className = "",
  buttonClassName = "",
}: CustomChatInputProps) {
  const [canSubmit, setCanSubmit] = React.useState(false);
  const {
    submittedMessage,
    clearAllFiles,
    setShouldHideSuggestions,
    setSubmittedMessage,
    images,
    files,
  } = useConversationStore();

  // Note: we intentionally do NOT disable the input when the conversation is
  // in an ERROR/STUCK execution state. Users should be able to send a follow-up
  // message to recover the conversation; the message will be delivered
  // immediately via the WebSocket if connected, or queued via REST otherwise.
  const isDisabled = disabled;

  // Always call the latest `onSubmit` without making the effect re-run when
  // its identity changes. `onSubmit` (typically `handleSendMessage`) is a
  // fresh function on every parent render, and the parent re-renders
  // whenever the pending-message queue updates synchronously inside
  // `onSubmit` itself. Listing it in the dep array caused the effect to
  // fire twice — once for the original submit and again from the
  // mid-submit re-render, before `setSubmittedMessage(null)` was applied —
  // producing a duplicate "Sending…" bubble.
  const onSubmitRef = useRef(onSubmit);
  useEffect(() => {
    onSubmitRef.current = onSubmit;
  }, [onSubmit]);

  // Listen to submittedMessage state changes
  useEffect(() => {
    if (!submittedMessage || disabled) {
      return;
    }
    onSubmitRef.current(submittedMessage);
    setSubmittedMessage(null);
  }, [submittedMessage, disabled, setSubmittedMessage]);

  // Custom hooks
  const {
    chatInputRef,
    messageToSend,
    checkIsContentEmpty,
    clearEmptyContentHandler,
    saveDraft,
  } = useChatInputLogic();

  const syncCanSubmit = React.useCallback(() => {
    const text = chatInputRef.current?.innerText ?? "";
    const hasAttachments = images.length > 0 || files.length > 0;
    setCanSubmit(text.trim().length > 0 || hasAttachments);
  }, [chatInputRef, images, files]);

  const {
    fileInputRef,
    chatContainerRef,
    isDragOver,
    handleFileIconClick,
    handleFileInputChange,
    handleDragOver,
    handleDragLeave,
    handleDrop,
  } = useFileHandling(onFilesPaste);

  const {
    gripRef,
    isGripVisible,
    isGripDragging,
    handleTopEdgeClick,
    smartResize,
    handleGripMouseDown,
    handleGripTouchStart,
    increaseHeightForEmptyContent,
    resetManualResize,
  } = useGripResize(
    chatInputRef as React.RefObject<HTMLDivElement | null>,
    messageToSend,
  );

  const { handleSubmit } = useChatSubmission(
    chatInputRef as React.RefObject<HTMLDivElement | null>,
    fileInputRef as React.RefObject<HTMLInputElement | null>,
    smartResize,
    onSubmit,
    resetManualResize,
  );
  const handleSubmitAndSync = React.useCallback(() => {
    handleSubmit();
    syncCanSubmit();
  }, [handleSubmit, syncCanSubmit]);

  const { handleInput, handlePaste, handleKeyDown, handleBlur, handleFocus } =
    useChatInputEvents(
      chatInputRef as React.RefObject<HTMLDivElement | null>,
      smartResize,
      increaseHeightForEmptyContent,
      checkIsContentEmpty,
      clearEmptyContentHandler,
      onFocus,
      onBlur,
    );

  const {
    isMenuOpen: isSlashMenuOpen,
    filteredItems: slashItems,
    selectedIndex: slashSelectedIndex,
    updateSlashMenu,
    selectItem: selectSlashItem,
    handleSlashKeyDown,
    closeMenu: closeSlashMenu,
  } = useSlashCommand(chatInputRef as React.RefObject<HTMLDivElement | null>);

  // Cleanup: reset suggestions visibility when component unmounts
  useEffect(
    () => () => {
      setShouldHideSuggestions(false);
      clearAllFiles();
    },
    [setShouldHideSuggestions, clearAllFiles],
  );
  useEffect(() => {
    syncCanSubmit();
  }, [syncCanSubmit, images.length, files.length]);
  return (
    <div className={cn("w-full", className)}>
      {/* Hidden file input */}
      <HiddenFileInput
        fileInputRef={fileInputRef}
        onChange={handleFileInputChange}
      />

      {/* Container with grip */}
      <div className="relative w-full">
        <ChatInputGrip
          gripRef={gripRef}
          isGripVisible={isGripVisible}
          isGripDragging={isGripDragging}
          handleTopEdgeClick={handleTopEdgeClick}
          handleGripMouseDown={handleGripMouseDown}
          handleGripTouchStart={handleGripTouchStart}
        />

        <ChatInputContainer
          chatContainerRef={chatContainerRef}
          isDragOver={isDragOver}
          disabled={isDisabled}
          canSubmit={canSubmit}
          hasStartedConversation={hasStartedConversation}
          isNewConversationPending={isNewConversationPending}
          showButton={showButton}
          buttonClassName={buttonClassName}
          chatInputRef={chatInputRef}
          handleFileIconClick={handleFileIconClick}
          handleSubmit={handleSubmitAndSync}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onInput={() => {
            handleInput();
            updateSlashMenu();
            saveDraft();
            syncCanSubmit();
          }}
          onPaste={handlePaste}
          onKeyDown={(e) => {
            if (handleSlashKeyDown(e)) return;
            handleKeyDown(e, isDisabled, handleSubmitAndSync);
          }}
          onFocus={handleFocus}
          onBlur={() => {
            handleBlur();
            closeSlashMenu();
            syncCanSubmit();
          }}
          isSlashMenuOpen={isSlashMenuOpen}
          slashItems={slashItems}
          slashSelectedIndex={slashSelectedIndex}
          onSlashSelect={selectSlashItem}
        />
      </div>
    </div>
  );
}
