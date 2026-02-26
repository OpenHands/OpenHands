import { useTranslation } from "react-i18next";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { useBreakpoint } from "#/hooks/use-breakpoint";
import { cn } from "#/utils/utils";
import { ContextMenu } from "#/ui/context-menu";
import { ContextMenuListItem } from "../context-menu/context-menu-list-item";
import { Divider } from "#/ui/divider";
import { I18nKey } from "#/i18n/declaration";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useConfig } from "#/hooks/query/use-config";

import EditIcon from "#/icons/u-edit.svg?react";
import RobotIcon from "#/icons/u-robot.svg?react";
import ToolsIcon from "#/icons/u-tools.svg?react";
import FileExportIcon from "#/icons/u-file-export.svg?react";
import DownloadIcon from "#/icons/u-download.svg?react";
import CreditCardIcon from "#/icons/u-credit-card.svg?react";
import CloseIcon from "#/icons/u-close.svg?react";
import DeleteIcon from "#/icons/u-delete.svg?react";
import LinkIcon from "#/icons/link-external.svg?react";
import CopyIcon from "#/icons/copy.svg?react";
import { ConversationNameContextMenuIconText } from "./conversation-name-context-menu-icon-text";
import { CONTEXT_MENU_ICON_TEXT_CLASSNAME } from "#/utils/constants";

const contextMenuListItemClassName = cn(
  "cursor-pointer p-0 h-auto hover:bg-transparent",
  CONTEXT_MENU_ICON_TEXT_CLASSNAME,
);

interface ConversationNameContextMenuProps {
  onClose: () => void;
  onRename?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onDelete?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onStop?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onDisplayCost?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onShowAgentTools?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onShowSkills?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onExportConversation?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onDownloadViaVSCode?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onTogglePublic?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onDownloadConversation?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onCopyShareLink?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  shareUrl?: string;
  position?: "top" | "bottom";
}

export function ConversationNameContextMenu({
  onClose,
  onRename,
  onDelete,
  onStop,
  onDisplayCost,
  onShowAgentTools,
  onShowSkills,
  onExportConversation,
  onDownloadViaVSCode,
  onTogglePublic,
  onDownloadConversation,
  onCopyShareLink,
  shareUrl,
  position = "bottom",
}: ConversationNameContextMenuProps) {
  const isMobile = useBreakpoint();

  const { t } = useTranslation();
  const ref = useClickOutsideElement<HTMLUListElement>(onClose);
  const { data: conversation } = useActiveConversation();
  const { data: config } = useConfig();

  // This is a temporary measure and may be re-enabled in the future
  const isV1Conversation = conversation?.conversation_version === "V1";

  // Check if we should show the public sharing option
  // Only show for V1 conversations in SAAS mode
  const shouldShowPublicSharing =
    isV1Conversation && config?.app_mode === "saas" && onTogglePublic;

  const hasDownload = Boolean(onDownloadViaVSCode || onDownloadConversation);
  const hasExport = Boolean(onExportConversation);
  const hasTools = Boolean(onShowAgentTools || onShowSkills);
  const hasInfo = Boolean(onDisplayCost);
  const hasControl = Boolean(onStop || onDelete);

  return (
    <ContextMenu
      ref={ref}
      testId="conversation-name-context-menu"
      position={position}
      alignment="left"
      className={isMobile ? "right-0 translate-x-[34%] left-auto" : ""}
    >
      {onRename && (
        <ContextMenuListItem
          testId="rename-button"
          onClick={onRename}
          className={contextMenuListItemClassName}
        >
          <ConversationNameContextMenuIconText
            icon={<EditIcon width={16} height={16} />}
            text={t(I18nKey.BUTTON$RENAME)}
            className={CONTEXT_MENU_ICON_TEXT_CLASSNAME}
          />
        </ContextMenuListItem>
      )}

      {hasTools && <Divider testId="separator-tools" />}

      {onShowSkills && (
        <ContextMenuListItem
          testId="show-skills-button"
          onClick={onShowSkills}
          className={contextMenuListItemClassName}
        >
          <ConversationNameContextMenuIconText
            icon={<RobotIcon width={16} height={16} />}
            text={t(I18nKey.CONVERSATION$SHOW_SKILLS)}
            className={CONTEXT_MENU_ICON_TEXT_CLASSNAME}
          />
        </ContextMenuListItem>
      )}

      {onShowAgentTools && (
        <ContextMenuListItem
          testId="show-agent-tools-button"
          onClick={onShowAgentTools}
          className={contextMenuListItemClassName}
        >
          <ConversationNameContextMenuIconText
            icon={<ToolsIcon width={16} height={16} />}
            text={t(I18nKey.BUTTON$SHOW_AGENT_TOOLS_AND_METADATA)}
            className={CONTEXT_MENU_ICON_TEXT_CLASSNAME}
          />
        </ContextMenuListItem>
      )}

      {(hasExport || hasDownload) && !isV1Conversation ? (
        <Divider testId="separator-export" />
      ) : null}

      {onExportConversation && !isV1Conversation && (
        <ContextMenuListItem
          testId="export-conversation-button"
          onClick={onExportConversation}
          className={contextMenuListItemClassName}
        >
          <ConversationNameContextMenuIconText
            icon={<FileExportIcon width={16} height={16} />}
            text={t(I18nKey.BUTTON$EXPORT_CONVERSATION)}
            className={CONTEXT_MENU_ICON_TEXT_CLASSNAME}
          />
        </ContextMenuListItem>
      )}

      {onDownloadViaVSCode && !isV1Conversation && (
        <ContextMenuListItem
          testId="download-vscode-button"
          onClick={onDownloadViaVSCode}
          className={contextMenuListItemClassName}
        >
          <ConversationNameContextMenuIconText
            icon={<DownloadIcon width={16} height={16} />}
            text={t(I18nKey.BUTTON$DOWNLOAD_VIA_VSCODE)}
            className={CONTEXT_MENU_ICON_TEXT_CLASSNAME}
          />
        </ContextMenuListItem>
      )}

      {onDownloadConversation && isV1Conversation && (
        <ContextMenuListItem
          testId="download-trajectory-button"
          onClick={onDownloadConversation}
          className={contextMenuListItemClassName}
        >
          <ConversationNameContextMenuIconText
            icon={<DownloadIcon width={16} height={16} />}
            text={t(I18nKey.BUTTON$EXPORT_CONVERSATION)}
            className={CONTEXT_MENU_ICON_TEXT_CLASSNAME}
          />
        </ContextMenuListItem>
      )}

      {(hasInfo || hasControl) && <Divider testId="separator-info-control" />}

      {onDisplayCost && (
        <ContextMenuListItem
          testId="display-cost-button"
          onClick={onDisplayCost}
          className={contextMenuListItemClassName}
        >
          <ConversationNameContextMenuIconText
            icon={<CreditCardIcon width={16} height={16} />}
            text={t(I18nKey.BUTTON$DISPLAY_COST)}
            className={CONTEXT_MENU_ICON_TEXT_CLASSNAME}
          />
        </ContextMenuListItem>
      )}

      {shouldShowPublicSharing && (
        <ContextMenuListItem
          testId="share-publicly-button"
          onClick={onTogglePublic}
          className={contextMenuListItemClassName}
        >
          <div className="flex items-center gap-2 justify-between w-full hover:bg-[#5C5D62] rounded h-[30px]">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={conversation?.public || false}
                className="w-4 h-4 ml-2 cursor-pointer"
              />
              <span>{t(I18nKey.CONVERSATION$SHARE_PUBLICLY)}</span>
            </div>
            {conversation?.public && shareUrl && onCopyShareLink && (
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  data-testid="copy-share-link-button"
                  onClick={onCopyShareLink}
                  className="p-1 hover:bg-[#717888] rounded cursor-pointer"
                  title={t(I18nKey.BUTTON$COPY_TO_CLIPBOARD)}
                >
                  <CopyIcon width={16} height={16} />
                </button>
                <a
                  data-testid="open-share-link-button"
                  href={shareUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                  className="p-1 hover:bg-[#717888] rounded cursor-pointer"
                  title={t(I18nKey.BUTTON$OPEN_IN_NEW_TAB)}
                >
                  <LinkIcon width={16} height={16} />
                </a>
              </div>
            )}
          </div>
        </ContextMenuListItem>
      )}

      {onStop && (
        <ContextMenuListItem
          testId="stop-button"
          onClick={onStop}
          className={contextMenuListItemClassName}
        >
          <ConversationNameContextMenuIconText
            icon={<CloseIcon width={16} height={16} />}
            text={t(I18nKey.COMMON$CLOSE_CONVERSATION_STOP_RUNTIME)}
            className={CONTEXT_MENU_ICON_TEXT_CLASSNAME}
          />
        </ContextMenuListItem>
      )}

      {onDelete && (
        <ContextMenuListItem
          testId="delete-button"
          onClick={onDelete}
          className={contextMenuListItemClassName}
        >
          <ConversationNameContextMenuIconText
            icon={<DeleteIcon width={16} height={16} />}
            text={t(I18nKey.COMMON$DELETE_CONVERSATION)}
            className={CONTEXT_MENU_ICON_TEXT_CLASSNAME}
          />
        </ContextMenuListItem>
      )}
    </ContextMenu>
  );
}
