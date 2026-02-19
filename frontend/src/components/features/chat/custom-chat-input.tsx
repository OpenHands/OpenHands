import React, { useState, useEffect, useCallback } from "react";
import { useTranslation } from "react-i18next";
import { ConversationStatus } from "#/types/conversation-status";
import { useChatInputLogic } from "#/hooks/chat/use-chat-input-logic";
import { useFileHandling } from "#/hooks/chat/use-file-handling";
import { useGripResize } from "#/hooks/chat/use-grip-resize";
import { useChatInputEvents } from "#/hooks/chat/use-chat-input-events";
import { useChatSubmission } from "#/hooks/chat/use-chat-submission";
import { useSlashCommands } from "#/hooks/chat/use-slash-commands";
import { useSlashCommandActions } from "#/hooks/chat/use-slash-command-actions";
import { ChatInputGrip } from "./components/chat-input-grip";
import { ChatInputContainer } from "./components/chat-input-container";
import { HiddenFileInput } from "./components/hidden-file-input";
import { SlashCommandMenu } from "./slash-command-menu";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { useConversationStore } from "#/stores/conversation-store";
import { I18nKey } from "#/i18n/declaration";
import { SlashCommand } from "#/config/slash-commands";

export interface CustomChatInputProps {
  disabled?: boolean;
  showButton?: boolean;
  conversationStatus?: ConversationStatus | null;
  onSubmit: (message: string) => void;
  onFocus?: () => void;
  onBlur?: () => void;
  onFilesPaste?: (files: File[]) => void;
  className?: React.HTMLAttributes<HTMLDivElement>["className"];
  buttonClassName?: React.HTMLAttributes<HTMLButtonElement>["className"];
}

export function CustomChatInput({
  disabled = false,
  showButton = true,
  conversationStatus = null,
  onSubmit,
  onFocus,
  onBlur,
  onFilesPaste,
  className = "",
  buttonClassName = "",
}: CustomChatInputProps) {
  const { t } = useTranslation();
  const {
    submittedMessage,
    clearAllFiles,
    setShouldHideSuggestions,
    setSubmittedMessage,
  } = useConversationStore();
  const [pendingClearCommand, setPendingClearCommand] =
    useState<SlashCommand | null>(null);

  // Disable input when conversation is stopped
  const isConversationStopped = conversationStatus === "STOPPED";
  const isDisabled = disabled || isConversationStopped;

  // Listen to submittedMessage state changes
  useEffect(() => {
    if (!submittedMessage || disabled) {
      return;
    }
    onSubmit(submittedMessage);
    setSubmittedMessage(null);
  }, [submittedMessage, disabled, onSubmit, setSubmittedMessage]);

  // Custom hooks
  const {
    chatInputRef,
    messageToSend,
    checkIsContentEmpty,
    clearEmptyContentHandler,
  } = useChatInputLogic();

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

  const { handleSubmit, handleResumeAgent } = useChatSubmission(
    chatInputRef as React.RefObject<HTMLDivElement | null>,
    fileInputRef as React.RefObject<HTMLInputElement | null>,
    smartResize,
    onSubmit,
    resetManualResize,
  );

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

  // Slash command hooks
  const {
    isMenuOpen,
    filteredCommands,
    selectedIndex,
    handleInputChange: handleSlashInputChange,
    handleKeyDown: handleSlashKeyDown,
    selectCommand,
    resetState: resetSlashState,
  } = useSlashCommands();

  const { executeCommand } = useSlashCommandActions();

  // Wrap executeCommand to require confirmation for destructive commands
  const executeOrConfirm = useCallback(
    (command: SlashCommand): boolean => {
      if (command.action === "clear") {
        setPendingClearCommand(command);
        return true;
      }
      return executeCommand(command);
    },
    [executeCommand],
  );

  // Handle input changes to sync with slash command state
  const handleInputWithSlash = useCallback(() => {
    handleInput();
    // Sync the input value with slash command hook
    const text = chatInputRef.current?.innerText || "";
    handleSlashInputChange(text);
  }, [handleInput, handleSlashInputChange, chatInputRef]);

  // Handle selecting a command from the menu
  const handleSelectCommand = useCallback(
    (command: (typeof filteredCommands)[number]) => {
      selectCommand(command);
      executeOrConfirm(command);
      resetSlashState();
      // Clear the input
      if (chatInputRef.current) {
        chatInputRef.current.textContent = "";
      }
      smartResize();
    },
    [selectCommand, executeOrConfirm, resetSlashState, chatInputRef, smartResize],
  );

  // Handle keyboard events with slash command support
  const handleKeyDownWithSlash = useCallback(
    (e: React.KeyboardEvent) => {
      // First check if slash command menu should handle the event
      const currentText = chatInputRef.current?.innerText || "";
      if (currentText.startsWith("/")) {
        const selectedCommand = handleSlashKeyDown(e);
        if (selectedCommand) {
          // Command was selected via Enter key
          executeOrConfirm(selectedCommand);
          resetSlashState();
          if (chatInputRef.current) {
            chatInputRef.current.textContent = "";
          }
          smartResize();
          return;
        }
        // If slash command hook handled the event (arrow keys, escape, tab)
        if (e.defaultPrevented) {
          return;
        }
      }

      // Otherwise, use the normal key handler
      handleKeyDown(e, isDisabled, handleSubmit);
    },
    [
      chatInputRef,
      handleSlashKeyDown,
      executeOrConfirm,
      resetSlashState,
      smartResize,
      handleKeyDown,
      isDisabled,
      handleSubmit,
    ],
  );

  // Cleanup: reset suggestions visibility when component unmounts
  useEffect(
    () => () => {
      setShouldHideSuggestions(false);
      clearAllFiles();
    },
    [setShouldHideSuggestions, clearAllFiles],
  );
  return (
    <div className={`w-full ${className}`}>
      {pendingClearCommand && (
        <ConfirmationModal
          text={t(I18nKey.SLASH_COMMANDS$CLEAR_CONFIRM)}
          onConfirm={() => {
            executeCommand(pendingClearCommand);
            setPendingClearCommand(null);
          }}
          onCancel={() => setPendingClearCommand(null)}
        />
      )}

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
          handleTopEdgeClick={handleTopEdgeClick}
          handleGripMouseDown={handleGripMouseDown}
          handleGripTouchStart={handleGripTouchStart}
        />

        {/* Slash command menu */}
        <SlashCommandMenu
          isOpen={isMenuOpen}
          commands={filteredCommands}
          selectedIndex={selectedIndex}
          onSelectCommand={handleSelectCommand}
        />

        <ChatInputContainer
          chatContainerRef={chatContainerRef}
          isDragOver={isDragOver}
          disabled={isDisabled}
          showButton={showButton}
          buttonClassName={buttonClassName}
          chatInputRef={chatInputRef}
          handleFileIconClick={handleFileIconClick}
          handleSubmit={handleSubmit}
          handleResumeAgent={handleResumeAgent}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onInput={handleInputWithSlash}
          onPaste={handlePaste}
          onKeyDown={handleKeyDownWithSlash}
          onFocus={handleFocus}
          onBlur={handleBlur}
        />
      </div>
    </div>
  );
}
