import React, { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { ContextMenu } from "#/ui/context-menu";
import { ContextMenuListItem } from "../../context-menu/context-menu-list-item";
import { I18nKey } from "#/i18n/declaration";
import { ConversationNameContextMenuIconText } from "../../conversation/conversation-name-context-menu-icon-text";

import EditIcon from "#/icons/u-edit.svg?react";
import SkillsIcon from "#/icons/skills.svg?react";
import ToolsIcon from "#/icons/u-tools.svg?react";
import DownloadIcon from "#/icons/u-download.svg?react";
import CreditCardIcon from "#/icons/u-credit-card.svg?react";
import CloseIcon from "#/icons/u-close.svg?react";
import DeleteIcon from "#/icons/u-delete.svg?react";
import { Divider } from "#/ui/divider";

interface ConversationCardContextMenuProps {
  onClose: () => void;
  onDelete?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onStop?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onEdit?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onDisplayCost?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onShowAgentTools?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onShowSkills?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onDownloadViaVSCode?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onDownloadConversation?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  position?: "top" | "bottom";
  /**
   * Fixed coordinates for a portaled menu (conversation list overflow
   * stacking). When set, theme/position switch to non-absolute layout.
   */
  floatingStyle?: React.CSSProperties;
  ignoreOutsideClickRef?: React.RefObject<HTMLElement | null>;
}

export function ConversationCardContextMenu({
  onClose,
  onDelete,
  onStop,
  onEdit,
  onDisplayCost,
  onShowAgentTools,
  onShowSkills,
  onDownloadViaVSCode,
  onDownloadConversation,
  position = "bottom",
  floatingStyle,
  ignoreOutsideClickRef,
}: ConversationCardContextMenuProps) {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  const ref = useClickOutsideElement<HTMLUListElement>(
    onClose,
    ignoreOutsideClickRef,
  );
  const stopLabelKey =
    backend.kind === "cloud"
      ? I18nKey.COMMON$CLOSE_CONVERSATION_STOP_RUNTIME
      : I18nKey.COMMON$STOP_CONVERSATION;

  const generateSection = useCallback(
    (items: React.ReactNode[], sectionKey: string, isLast?: boolean) => {
      const filteredItems = items.filter((i) => i != null);

      if (filteredItems.length > 0) {
        return !isLast ? (
          <React.Fragment key={sectionKey}>
            {filteredItems}
            <Divider inset="menu" />
          </React.Fragment>
        ) : (
          <React.Fragment key={sectionKey}>{filteredItems}</React.Fragment>
        );
      }
      return null;
    },
    [],
  );

  const isPortaled = floatingStyle != null;

  return (
    <ContextMenu
      ref={ref}
      testId="context-menu"
      style={isPortaled ? floatingStyle : undefined}
      theme={isPortaled ? "popover" : "default"}
      position={isPortaled ? "none" : position}
      alignment={isPortaled ? "none" : "right"}
      spacing={isPortaled ? "none" : "default"}
      className={
        isPortaled
          ? "mt-0 min-w-[200px] w-max max-w-[min(280px,100vw-16px)]"
          : "z-[200] mt-0"
      }
    >
      {generateSection(
        [
          onEdit && (
            <ContextMenuListItem
              key="edit-button"
              testId="edit-button"
              onClick={onEdit}
            >
              <ConversationNameContextMenuIconText
                icon={<EditIcon width={16} height={16} />}
                text={t(I18nKey.BUTTON$RENAME)}
              />
            </ContextMenuListItem>
          ),
        ],
        // eslint-disable-next-line i18next/no-literal-string -- internal section id, not user-facing
        "edit-section",
      )}
      {generateSection(
        [
          onShowAgentTools && (
            <ContextMenuListItem
              key="show-agent-tools-button"
              testId="show-agent-tools-button"
              onClick={onShowAgentTools}
            >
              <ConversationNameContextMenuIconText
                icon={<ToolsIcon width={16} height={16} />}
                text={t(I18nKey.BUTTON$SHOW_AGENT_TOOLS_AND_METADATA)}
              />
            </ContextMenuListItem>
          ),
          onShowSkills && (
            <ContextMenuListItem
              key="show-skills-button"
              testId="show-skills-button"
              onClick={onShowSkills}
            >
              <ConversationNameContextMenuIconText
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
          ),
        ],
        // eslint-disable-next-line i18next/no-literal-string -- internal section id, not user-facing
        "tools-section",
      )}
      {generateSection(
        [
          onStop && (
            <ContextMenuListItem
              key="stop-button"
              testId="stop-button"
              onClick={onStop}
            >
              <ConversationNameContextMenuIconText
                icon={<CloseIcon width={16} height={16} />}
                text={t(stopLabelKey)}
              />
            </ContextMenuListItem>
          ),
          onDownloadViaVSCode && (
            <ContextMenuListItem
              key="download-vscode-button"
              testId="download-vscode-button"
              onClick={onDownloadViaVSCode}
            >
              <ConversationNameContextMenuIconText
                icon={<DownloadIcon width={16} height={16} />}
                text={t(I18nKey.BUTTON$DOWNLOAD_VIA_VSCODE)}
              />
            </ContextMenuListItem>
          ),
          onDownloadConversation && (
            <ContextMenuListItem
              key="download-trajectory-button"
              testId="download-trajectory-button"
              onClick={onDownloadConversation}
            >
              <ConversationNameContextMenuIconText
                icon={<DownloadIcon width={16} height={16} />}
                text={t(I18nKey.BUTTON$DOWNLOAD_CONVERSATION_DATA)}
              />
            </ContextMenuListItem>
          ),
        ],
        // eslint-disable-next-line i18next/no-literal-string -- internal section id, not user-facing
        "control-section",
      )}
      {generateSection(
        [
          onDisplayCost && (
            <ContextMenuListItem
              key="display-cost-button"
              testId="display-cost-button"
              onClick={onDisplayCost}
            >
              <ConversationNameContextMenuIconText
                icon={<CreditCardIcon width={16} height={16} />}
                text={t(I18nKey.BUTTON$DISPLAY_COST)}
              />
            </ContextMenuListItem>
          ),
          onDelete && (
            <ContextMenuListItem
              key="delete-button"
              testId="delete-button"
              onClick={onDelete}
            >
              <ConversationNameContextMenuIconText
                icon={<DeleteIcon width={16} height={16} />}
                text={t(I18nKey.COMMON$DELETE_CONVERSATION)}
              />
            </ContextMenuListItem>
          ),
        ],
        // eslint-disable-next-line i18next/no-literal-string -- internal section id, not user-facing
        "info-section",
        true,
      )}
    </ContextMenu>
  );
}
