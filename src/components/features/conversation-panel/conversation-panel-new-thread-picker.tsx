import { useTranslation } from "react-i18next";
import { FolderPlus } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import type { BackendKind } from "#/api/backend-registry/types";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { cn } from "#/utils/utils";
import { LocalNewConversationMenu } from "./local-new-conversation-menu";
import { CloudNewConversationMenu } from "./cloud-new-conversation-menu";

const triggerClassName = cn(
  "inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md",
  "text-[var(--oh-muted)] transition-colors",
  "hover:bg-[var(--oh-surface-raised)] hover:text-white",
  "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--oh-border)]",
  "disabled:cursor-not-allowed disabled:opacity-50",
);

/**
 * Compact "new thread folder" control for the conversation panel header.
 * Opens the same workspace (local) or repository (cloud) picker as the
 * sidebar new-conversation flows.
 */
export function ConversationPanelNewThreadPicker({
  backendKind,
}: {
  backendKind: BackendKind;
}) {
  const { t } = useTranslation("openhands");
  const ariaLabel = t(I18nKey.CONVERSATION_PANEL$NEW_THREAD_FOLDER_ARIA);

  const triggerIcon = (
    <FolderPlus className="h-4 w-4 shrink-0" aria-hidden strokeWidth={2} />
  );

  if (backendKind === "local") {
    return (
      <LocalNewConversationMenu
        useFixedPlacement
        popoverTestId="conversation-panel-new-thread-popover"
        popoverClassName=""
        trigger={({
          onClick,
          "aria-expanded": expanded,
          disabled,
          "aria-haspopup": hasPopup,
        }) => (
          <StyledTooltip content={ariaLabel} placement="bottom">
            <button
              type="button"
              className={triggerClassName}
              aria-label={ariaLabel}
              aria-expanded={expanded}
              aria-haspopup={hasPopup}
              disabled={disabled}
              data-testid="conversation-panel-new-thread-picker"
              onClick={onClick}
            >
              {triggerIcon}
            </button>
          </StyledTooltip>
        )}
      />
    );
  }

  return (
    <CloudNewConversationMenu
      useFixedPlacement
      popoverTestId="conversation-panel-new-thread-popover"
      popoverClassName=""
      trigger={({
        onClick,
        "aria-expanded": expanded,
        disabled,
        "aria-haspopup": hasPopup,
      }) => (
        <StyledTooltip content={ariaLabel} placement="bottom">
          <button
            type="button"
            className={triggerClassName}
            aria-label={ariaLabel}
            aria-expanded={expanded}
            aria-haspopup={hasPopup}
            disabled={disabled}
            data-testid="conversation-panel-new-thread-picker"
            onClick={onClick}
          >
            {triggerIcon}
          </button>
        </StyledTooltip>
      )}
    />
  );
}
