import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useConversationStore } from "#/stores/conversation-store";

interface ChatInputFieldProps {
  chatInputRef: React.RefObject<HTMLDivElement | null>;
  onInput: () => void;
  onPaste: (e: React.ClipboardEvent) => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
  onFocus?: () => void;
  onBlur?: () => void;
}

export function ChatInputField({
  chatInputRef,
  onInput,
  onPaste,
  onKeyDown,
  onFocus,
  onBlur,
}: ChatInputFieldProps) {
  const { t } = useTranslation();

  const conversationMode = useConversationStore(
    (state) => state.conversationMode,
  );

  const isPlanMode = conversationMode === "plan";

  return (
    <div
      className="box-border content-stretch flex flex-row items-center justify-start min-h-6 p-0 relative shrink-0 flex-1"
      data-name="Text & caret"
    >
      <div className="basis-0 flex flex-col font-normal grow justify-center leading-[0] min-h-px min-w-px overflow-ellipsis overflow-hidden relative shrink-0 text-[#d0d9fa] text-[16px] text-left">
        <div
          ref={chatInputRef}
          className="chat-input bg-transparent text-white text-[16px] font-normal leading-[20px] outline-none resize-none custom-scrollbar min-h-[20px] max-h-[400px] [text-overflow:inherit] [text-wrap-mode:inherit] [white-space-collapse:inherit] block whitespace-pre-wrap"
          contentEditable
          data-placeholder={
            isPlanMode
              ? t(I18nKey.COMMON$LET_S_WORK_ON_A_PLAN)
              : t(I18nKey.SUGGESTIONS$WHAT_TO_BUILD)
          }
          data-testid="chat-input"
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
