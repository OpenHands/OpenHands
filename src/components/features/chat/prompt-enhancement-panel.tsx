import React from "react";
import { Loader2, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import type {
  PromptEnhancementController,
  PromptEnhancementErrorReason,
} from "#/hooks/chat/use-prompt-enhancement";
import { I18nKey } from "#/i18n/declaration";
import {
  chatInputIconButtonClassName,
  formControlMultilineFieldClassName,
} from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";

const ERROR_MESSAGE_KEYS: Record<PromptEnhancementErrorReason, I18nKey> = {
  draft_changed: I18nKey.CHAT_INTERFACE$ENHANCE_PROMPT_ERROR_DRAFT_CHANGED,
  input_too_large: I18nKey.CHAT_INTERFACE$ENHANCE_PROMPT_ERROR_TOO_LARGE,
  timeout: I18nKey.CHAT_INTERFACE$ENHANCE_PROMPT_ERROR_TIMEOUT,
  invalid_output: I18nKey.CHAT_INTERFACE$ENHANCE_PROMPT_ERROR_INVALID_OUTPUT,
  profile_unavailable: I18nKey.CHAT_INTERFACE$ENHANCE_PROMPT_ERROR_UNAVAILABLE,
  unsupported: I18nKey.CHAT_INTERFACE$ENHANCE_PROMPT_ERROR_UNAVAILABLE,
  failed: I18nKey.CHAT_INTERFACE$ENHANCE_PROMPT_ERROR_FAILED,
};

interface PromptEnhancementPanelProps {
  enhancement: PromptEnhancementController;
}

/**
 * Review area above the composer: the loading state, an editable suggestion
 * with "Use enhanced prompt" / "Keep original" / cancel, or an error. Every
 * exit except "Use enhanced prompt" leaves the draft as it was.
 */
export function PromptEnhancementPanel({
  enhancement,
}: PromptEnhancementPanelProps) {
  const { t } = useTranslation("openhands");
  const { state, dismiss, setEnhancedText, accept } = enhancement;
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);
  const isPreview = state.status === "preview";

  // Move focus to the suggestion when it arrives, with the caret at the end,
  // so it can be read and edited right away.
  React.useEffect(() => {
    const textarea = textareaRef.current;
    if (!isPreview || !textarea) return;
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
  }, [isPreview]);

  if (state.status === "idle") return null;

  return (
    <section
      data-testid="prompt-enhancement-panel"
      aria-label={t(I18nKey.CHAT_INTERFACE$ENHANCED_PROMPT_TITLE)}
      className="mb-3 flex w-full flex-col gap-2 rounded-lg border border-border bg-surface-raised p-3"
    >
      {state.status === "loading" && (
        <div className="flex items-center justify-between gap-3">
          <span
            role="status"
            className="flex items-center gap-2 text-sm text-muted"
          >
            <Loader2 className="size-4 animate-spin" aria-hidden />
            {t(I18nKey.CHAT_INTERFACE$ENHANCE_PROMPT_LOADING)}
          </span>
          <BrandButton
            type="button"
            variant="secondary"
            testId="prompt-enhancement-cancel"
            onClick={dismiss}
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
        </div>
      )}

      {state.status === "error" && (
        <div className="flex items-center justify-between gap-3">
          <p role="alert" className="text-sm text-contrast">
            {t(ERROR_MESSAGE_KEYS[state.reason])}
          </p>
          <BrandButton
            type="button"
            variant="secondary"
            testId="prompt-enhancement-close"
            onClick={dismiss}
          >
            {t(I18nKey.BUTTON$CLOSE)}
          </BrandButton>
        </div>
      )}

      {state.status === "preview" && (
        <>
          <div className="flex items-start justify-between gap-2">
            <div className="flex flex-col gap-0.5">
              <span className="text-sm font-semibold text-contrast">
                {t(I18nKey.CHAT_INTERFACE$ENHANCED_PROMPT_TITLE)}
              </span>
              <span className="text-xs text-muted">
                {t(I18nKey.CHAT_INTERFACE$ENHANCED_PROMPT_HINT)}
              </span>
            </div>
            <button
              type="button"
              className={cn(chatInputIconButtonClassName, "size-6 shrink-0")}
              aria-label={t(I18nKey.BUTTON$CANCEL)}
              data-testid="prompt-enhancement-cancel"
              onClick={dismiss}
            >
              <X className="size-4" aria-hidden />
            </button>
          </div>
          <textarea
            ref={textareaRef}
            data-testid="prompt-enhancement-text"
            aria-label={t(I18nKey.CHAT_INTERFACE$ENHANCED_PROMPT_TITLE)}
            className={cn(
              formControlMultilineFieldClassName,
              "custom-scrollbar max-h-60 min-h-24 resize-y",
            )}
            rows={6}
            value={state.enhancedText}
            onChange={(event) => setEnhancedText(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                event.preventDefault();
                dismiss();
              }
            }}
          />
          <div className="flex flex-wrap justify-end gap-2">
            <BrandButton
              type="button"
              variant="secondary"
              testId="prompt-enhancement-keep-original"
              onClick={dismiss}
            >
              {t(I18nKey.CHAT_INTERFACE$KEEP_ORIGINAL_PROMPT)}
            </BrandButton>
            <BrandButton
              type="button"
              variant="primary"
              testId="prompt-enhancement-use"
              isDisabled={!state.enhancedText.trim()}
              onClick={accept}
            >
              {t(I18nKey.CHAT_INTERFACE$USE_ENHANCED_PROMPT)}
            </BrandButton>
          </div>
        </>
      )}
    </section>
  );
}
