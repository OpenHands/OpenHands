import React from "react";
import { useTranslation } from "react-i18next";
import { Plus } from "lucide-react";

import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { useActiveBackend } from "#/contexts/active-backend-context";

import {
  CloudNewConversationMenu,
  type CloudNewConversationMenuTriggerProps,
} from "./cloud-new-conversation-menu";
import {
  LocalNewConversationMenu,
  type LocalNewConversationMenuTriggerProps,
} from "./local-new-conversation-menu";

interface NewConversationButtonProps {
  /**
   * Render the trigger as a "+" icon-only button (used by the collapsed
   * sidebar). The popover content is unchanged; only the trigger pill
   * collapses.
   */
  compact?: boolean;
}

/**
 * Sidebar "+ New Conversation" trigger.
 *
 * The popover content depends on the active backend: local backends operate
 * on workspace folders so we surface the workspace picker, while cloud
 * backends operate on git repositories so we surface a repository picker.
 *
 * The trigger pill (the "+" button itself) is identical for both variants;
 * only the menu component differs. Both `Cloud/LocalNewConversationMenu`
 * accept structurally identical trigger props, so a single `renderTrigger`
 * callback can satisfy either menu's type signature.
 */
export function NewConversationButton({
  compact = false,
}: NewConversationButtonProps = {}) {
  const { t } = useTranslation("openhands");
  const newConversationLabel = t(I18nKey.SIDEBAR$NEW_CONVERSATION);
  const isCloud = useActiveBackend().backend.kind === "cloud";

  const renderTrigger = React.useCallback(
    (
      tp:
        | CloudNewConversationMenuTriggerProps
        | LocalNewConversationMenuTriggerProps,
    ) => {
      const triggerButton = (
        <button
          type="button"
          data-testid="new-conversation-button"
          {...tp}
          aria-label={compact ? newConversationLabel : undefined}
          className={cn(
            "flex items-center rounded-md cursor-pointer transition-colors",
            "text-sm text-white bg-[var(--oh-surface)]/60 hover:bg-[var(--oh-surface-raised)]",
            "border border-[var(--oh-border)]",
            compact
              ? "justify-center w-10 h-10 p-0 mx-auto"
              : "gap-1.5 w-full px-3 py-2",
          )}
        >
          <Plus width={16} height={16} className="shrink-0" />
          {!compact && newConversationLabel}
        </button>
      );

      return compact ? (
        <StyledTooltip content={newConversationLabel} placement="right">
          {triggerButton}
        </StyledTooltip>
      ) : (
        triggerButton
      );
    },
    [compact, newConversationLabel],
  );

  const wrapperClassName = cn(compact && "flex justify-center");
  const popoverClassName = compact ? "left-0 w-[260px]" : "left-0 right-0";

  return isCloud ? (
    <CloudNewConversationMenu
      className={wrapperClassName}
      popoverClassName={popoverClassName}
      trigger={renderTrigger}
    />
  ) : (
    <LocalNewConversationMenu
      className={wrapperClassName}
      popoverClassName={popoverClassName}
      trigger={renderTrigger}
    />
  );
}
