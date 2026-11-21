import { Loader2, WandSparkles } from "lucide-react";
import { useTranslation } from "react-i18next";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { I18nKey } from "#/i18n/declaration";
import type {
  PromptEnhancementAvailabilityState,
  PromptEnhancementUnavailableReason,
} from "#/hooks/query/use-prompt-enhancement-availability";
import { chatInputIconButtonClassName } from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";

const UNAVAILABLE_MESSAGE_KEYS: Record<
  PromptEnhancementUnavailableReason,
  I18nKey
> = {
  cloud_backend: I18nKey.CHAT_INTERFACE$ENHANCE_PROMPT_UNAVAILABLE_CLOUD,
  acp_agent: I18nKey.CHAT_INTERFACE$ENHANCE_PROMPT_UNAVAILABLE_ACP,
  no_profile: I18nKey.CHAT_INTERFACE$ENHANCE_PROMPT_UNAVAILABLE_NO_PROFILE,
  checking: I18nKey.CHAT_INTERFACE$ENHANCE_PROMPT_CHECKING,
  unsupported_backend: I18nKey.CHAT_INTERFACE$ENHANCE_PROMPT_UNAVAILABLE_SERVER,
  profile_unavailable:
    I18nKey.CHAT_INTERFACE$ENHANCE_PROMPT_UNAVAILABLE_PROFILE,
  unsupported_configuration:
    I18nKey.CHAT_INTERFACE$ENHANCE_PROMPT_UNAVAILABLE_PROFILE,
  check_failed: I18nKey.CHAT_INTERFACE$ENHANCE_PROMPT_UNAVAILABLE_ERROR,
};

export interface ChatEnhancePromptButtonProps {
  availability: PromptEnhancementAvailabilityState;
  /** The draft has text to enhance (attachments alone do not count). */
  hasDraftText: boolean;
  isEnhancing: boolean;
  onEnhance: () => void;
}

/**
 * Composer action that asks for an enhanced version of the draft. It stays
 * focusable when unavailable so the tooltip can explain why.
 */
export function ChatEnhancePromptButton({
  availability,
  hasDraftText,
  isEnhancing,
  onEnhance,
}: ChatEnhancePromptButtonProps) {
  const { t } = useTranslation("openhands");
  const label = t(I18nKey.CHAT_INTERFACE$ENHANCE_PROMPT);
  const isDisabled = !availability.isAvailable || !hasDraftText || isEnhancing;
  const Icon = isEnhancing ? Loader2 : WandSparkles;

  return (
    <StyledTooltip
      content={
        availability.isAvailable
          ? label
          : t(UNAVAILABLE_MESSAGE_KEYS[availability.reason])
      }
      placement="top"
    >
      <button
        type="button"
        className={cn(
          chatInputIconButtonClassName,
          "shrink-0 size-8",
          isDisabled && "cursor-not-allowed text-text-subtle",
        )}
        aria-label={label}
        aria-disabled={isDisabled}
        aria-busy={isEnhancing}
        data-testid="chat-enhance-prompt-button"
        data-unavailable-reason={
          availability.isAvailable ? undefined : availability.reason
        }
        onClick={() => {
          if (!isDisabled) onEnhance();
        }}
      >
        <Icon
          className={cn("size-4", isEnhancing && "animate-spin")}
          aria-hidden
        />
      </button>
    </StyledTooltip>
  );
}
