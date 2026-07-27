import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useUserProviders } from "#/hooks/use-user-providers";
import { cn } from "#/utils/utils";
import { ContextMenu } from "#/ui/context-menu";
import { ContextMenuListItem } from "../context-menu/context-menu-list-item";
import { Divider } from "#/ui/divider";
import { I18nKey } from "#/i18n/declaration";

import CodeBranchIcon from "#/icons/u-code-branch.svg?react";
import SkillsIcon from "#/icons/skills.svg?react";
import PuzzleIcon from "#/icons/u-puzzle-piece.svg?react";
import FishingHookIcon from "#/icons/fishing-hook.svg?react";
import ToolsIcon from "#/icons/u-tools.svg?react";
import SettingsIcon from "#/icons/settings.svg?react";
import CarretRightFillIcon from "#/icons/carret-right-fill.svg?react";
import { ToolsContextMenuIconText } from "./tools-context-menu-icon-text";
import { GitToolsSubmenu } from "./git-tools-submenu";
import { MacrosSubmenu } from "./macros-submenu";
import { ArchivedDisabledTooltip } from "../context-menu/archived-disabled-tooltip";
import { useIsArchivedConversation } from "#/hooks/use-is-archived-conversation";

interface ToolsContextMenuProps {
  onClose: () => void;
  onShowSkills: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onShowPlugins: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onShowHooks: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onShowAgentTools: (event: React.MouseEvent<HTMLButtonElement>) => void;
  shouldShowAgentTools?: boolean;
  shouldShowHooks?: boolean;
  shouldShowPlugins?: boolean;
  /** When set, renders a divider and this action as the last menu item. */
  footerAction?: {
    testId: string;
    icon: React.ReactNode;
    label: string;
    onClick: () => void;
  };
}

export function ToolsContextMenu({
  onClose,
  onShowSkills,
  onShowPlugins,
  onShowHooks,
  onShowAgentTools,
  shouldShowAgentTools = true,
  shouldShowHooks = false,
  shouldShowPlugins = false,
  footerAction,
}: ToolsContextMenuProps) {
  const { t } = useTranslation("openhands");
  const { data: conversation } = useActiveConversation();
  const { providers } = useUserProviders();
  const isArchivedConversation = useIsArchivedConversation();

  const [activeSubmenu, setActiveSubmenu] = useState<"git" | "macros" | null>(
    null,
  );

  const hasRepository = !!conversation?.selected_repository;
  const providersAreSet = providers.length > 0;
  const showGitTools = hasRepository && providersAreSet;

  const handleSubmenuClick = (submenu: "git" | "macros") => {
    if (isArchivedConversation) {
      return;
    }
    setActiveSubmenu(activeSubmenu === submenu ? null : submenu);
  };

  const handleClose = () => {
    setActiveSubmenu(null);
    onClose();
  };

  const ref = useClickOutsideElement<HTMLUListElement>(handleClose);

  return (
    <ContextMenu
      ref={ref}
      testId="tools-context-menu"
      position="top"
      alignment="left"
      className="left-[-16px] mb-2 bottom-full overflow-visible min-w-[200px]"
    >
      {/* Git Tools */}
      {showGitTools && (
        <div className="relative group/git">
          <ArchivedDisabledTooltip isDisabled={isArchivedConversation}>
            <ContextMenuListItem
              testId="git-tools-button"
              onClick={() => handleSubmenuClick("git")}
              isDisabled={isArchivedConversation}
            >
              <ToolsContextMenuIconText
                icon={<CodeBranchIcon width={16} height={16} />}
                text={t(I18nKey.COMMON$GIT_TOOLS)}
                rightIcon={<CarretRightFillIcon width={10} height={10} />}
              />
            </ContextMenuListItem>
          </ArchivedDisabledTooltip>
          {!isArchivedConversation && (
            <div
              className={cn(
                "absolute left-full top-[-6px] z-60 opacity-0 invisible pointer-events-none transition-all duration-200 ml-[1px]",
                "group-hover/git:opacity-100 group-hover/git:visible group-hover/git:pointer-events-auto",
                "hover:opacity-100 hover:visible hover:pointer-events-auto",
                activeSubmenu === "git" &&
                  "opacity-100 visible pointer-events-auto",
              )}
            >
              <GitToolsSubmenu onClose={handleClose} />
            </div>
          )}
        </div>
      )}

      {/* Macros */}
      <div className="relative group/macros">
        <ArchivedDisabledTooltip isDisabled={isArchivedConversation}>
          <ContextMenuListItem
            testId="macros-button"
            onClick={() => handleSubmenuClick("macros")}
            isDisabled={isArchivedConversation}
          >
            <ToolsContextMenuIconText
              icon={<SettingsIcon width={16} height={16} />}
              text={t(I18nKey.COMMON$MACROS)}
              rightIcon={<CarretRightFillIcon width={10} height={10} />}
            />
          </ContextMenuListItem>
        </ArchivedDisabledTooltip>
        {!isArchivedConversation && (
          <div
            className={cn(
              "absolute left-full top-[-4px] z-60 opacity-0 invisible pointer-events-none transition-all duration-200 ml-[1px]",
              "group-hover/macros:opacity-100 group-hover/macros:visible group-hover/macros:pointer-events-auto",
              "hover:opacity-100 hover:visible hover:pointer-events-auto",
              activeSubmenu === "macros" &&
                "opacity-100 visible pointer-events-auto",
            )}
          >
            <MacrosSubmenu onClose={handleClose} />
          </div>
        )}
      </div>

      {shouldShowAgentTools && <Divider inset="menu" />}

      <ArchivedDisabledTooltip isDisabled={isArchivedConversation}>
        <ContextMenuListItem
          testId="show-skills-button"
          onClick={onShowSkills}
          isDisabled={isArchivedConversation}
        >
          <ToolsContextMenuIconText
            icon={
              <SkillsIcon
                width={16}
                height={16}
                className="stroke-[1.75]"
                aria-hidden
              />
            }
            text={t(I18nKey.CONVERSATION$SHOW_SKILLS)}
          />
        </ContextMenuListItem>
      </ArchivedDisabledTooltip>

      {/* Show Plugins - only when this conversation has attached plugins */}
      {shouldShowPlugins && (
        <ArchivedDisabledTooltip isDisabled={isArchivedConversation}>
          <ContextMenuListItem
            testId="show-plugins-button"
            onClick={onShowPlugins}
            isDisabled={isArchivedConversation}
          >
            <ToolsContextMenuIconText
              icon={<PuzzleIcon width={16} height={16} aria-hidden />}
              text={t(I18nKey.CONVERSATION$SHOW_PLUGINS)}
            />
          </ContextMenuListItem>
        </ArchivedDisabledTooltip>
      )}

      {/* Show Hooks - Only show for V1 conversations */}
      {shouldShowHooks && (
        <ArchivedDisabledTooltip isDisabled={isArchivedConversation}>
          <ContextMenuListItem
            testId="show-hooks-button"
            onClick={onShowHooks}
            isDisabled={isArchivedConversation}
          >
            <ToolsContextMenuIconText
              icon={<FishingHookIcon width={16} height={16} aria-hidden />}
              text={t(I18nKey.CONVERSATION$SHOW_HOOKS)}
            />
          </ContextMenuListItem>
        </ArchivedDisabledTooltip>
      )}

      {/* Show Agent Tools and Metadata - Only show if system message is available */}
      {shouldShowAgentTools && (
        <ArchivedDisabledTooltip isDisabled={isArchivedConversation}>
          <ContextMenuListItem
            testId="show-agent-tools-button"
            onClick={onShowAgentTools}
            isDisabled={isArchivedConversation}
          >
            <ToolsContextMenuIconText
              icon={<ToolsIcon width={16} height={16} />}
              text={t(I18nKey.BUTTON$SHOW_AGENT_TOOLS_AND_METADATA)}
            />
          </ContextMenuListItem>
        </ArchivedDisabledTooltip>
      )}

      {footerAction && (
        <>
          <Divider />
          <ContextMenuListItem
            testId={footerAction.testId}
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              footerAction.onClick();
              handleClose();
            }}
          >
            <ToolsContextMenuIconText
              icon={footerAction.icon}
              text={footerAction.label}
            />
          </ContextMenuListItem>
        </>
      )}
    </ContextMenu>
  );
}
