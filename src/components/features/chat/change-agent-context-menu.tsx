import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import CodeTagIcon from "#/icons/code-tag.svg?react";
import LessonPlanIcon from "#/icons/lesson-plan.svg?react";
import { ContextMenu } from "#/ui/context-menu";
import { ContextMenuListItem } from "../context-menu/context-menu-list-item";
import { ContextMenuIconTextWithDescription } from "../context-menu/context-menu-icon-text-with-description";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { cn } from "#/utils/utils";
import type { ConversationMode } from "#/stores/conversation-store";

interface ChangeAgentContextMenuProps {
  activeMode: ConversationMode;
  onClose: () => void;
  onCodeClick?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onPlanClick?: (event: React.MouseEvent<HTMLButtonElement>) => void;
}

export function ChangeAgentContextMenu({
  activeMode,
  onClose,
  onCodeClick,
  onPlanClick,
}: ChangeAgentContextMenuProps) {
  const { t } = useTranslation("openhands");
  const menuRef = useClickOutsideElement<HTMLUListElement>(onClose);

  const handleCodeClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    onCodeClick?.(event);
    onClose();
  };

  const handlePlanClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    onPlanClick?.(event);
    onClose();
  };

  return (
    <ContextMenu
      ref={menuRef}
      testId="change-agent-context-menu"
      position="top"
      alignment="left"
      className="min-h-fit mb-2 min-w-[195px] max-w-[195px]"
    >
      <ContextMenuListItem
        testId="code-option"
        onClick={handleCodeClick}
        className={cn(
          activeMode === "code" && "bg-[var(--oh-interactive-hover)]",
        )}
      >
        <ContextMenuIconTextWithDescription
          icon={CodeTagIcon}
          title={t(I18nKey.COMMON$CODE)}
          description={t(I18nKey.COMMON$CODE_AGENT_DESCRIPTION)}
          isActive={activeMode === "code"}
        />
      </ContextMenuListItem>
      <ContextMenuListItem
        testId="plan-option"
        onClick={handlePlanClick}
        className={cn(
          activeMode === "plan" && "bg-[var(--oh-interactive-hover)]",
        )}
      >
        <ContextMenuIconTextWithDescription
          icon={LessonPlanIcon}
          title={t(I18nKey.COMMON$PLAN)}
          description={t(I18nKey.COMMON$PLAN_AGENT_DESCRIPTION)}
          isActive={activeMode === "plan"}
        />
      </ContextMenuListItem>
    </ContextMenu>
  );
}
