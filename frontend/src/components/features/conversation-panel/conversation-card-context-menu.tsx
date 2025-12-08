import {
  Trash,
  Power,
  Pencil,
  Download,
  Wallet,
  Wrench,
  Bot,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { cn } from "#/utils/utils";
import { ContextMenu } from "#/ui/context-menu";
import { ContextMenuListItem } from "../context-menu/context-menu-list-item";
import { Divider } from "#/ui/divider";
import { I18nKey } from "#/i18n/declaration";
import { ContextMenuIconText } from "../context-menu/context-menu-icon-text";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";

interface ConversationCardContextMenuProps {
  onClose: () => void;
  onDelete?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onStop?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onEdit?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onDisplayCost?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onShowAgentTools?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onShowMicroagents?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onDownloadViaVSCode?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  position?: "top" | "bottom";
}

export function ConversationCardContextMenu({
  onClose,
  onDelete,
  onStop,
  onEdit,
  onDisplayCost,
  onShowAgentTools,
  onShowMicroagents,
  onDownloadViaVSCode,
  position = "bottom",
}: ConversationCardContextMenuProps) {
  const { t } = useTranslation();
  const ref = useClickOutsideElement<HTMLUListElement>(onClose);
  const { data: conversation } = useActiveConversation();

  // TODO: Hide microagent menu items for V1 conversations
  // This is a temporary measure and may be re-enabled in the future
  const isV1Conversation = conversation?.conversation_version === "V1";

  const hasEdit = Boolean(onEdit);
  const hasDownload = Boolean(onDownloadViaVSCode);
  const hasTools = Boolean(onShowAgentTools || onShowMicroagents);
  const hasInfo = Boolean(onDisplayCost);
  const hasControl = Boolean(onStop || onDelete);

  return (
    <ContextMenu
      ref={ref}
      testId="context-menu"
      className={cn(
        "right-0 absolute mt-3",
        position === "top" && "bottom-full",
        position === "bottom" && "top-full",
      )}
    >
      {onEdit && (
        <ContextMenuListItem testId="edit-button" onClick={onEdit}>
          <ContextMenuIconText
            icon={Pencil}
            text={t(I18nKey.BUTTON$EDIT_TITLE)}
          />
        </ContextMenuListItem>
      )}

      {hasEdit && (hasDownload || hasTools || hasInfo || hasControl) && (
        <Divider />
      )}

      {onDownloadViaVSCode && (
        <ContextMenuListItem
          testId="download-vscode-button"
          onClick={onDownloadViaVSCode}
        >
          <ContextMenuIconText
            icon={Download}
            text={t(I18nKey.BUTTON$DOWNLOAD_VIA_VSCODE)}
          />
        </ContextMenuListItem>
      )}

      {hasDownload && (hasTools || hasInfo || hasControl) && <Divider />}

      {onShowAgentTools && (
        <ContextMenuListItem
          testId="show-agent-tools-button"
          onClick={onShowAgentTools}
        >
          <ContextMenuIconText
            icon={Wrench}
            text={t(I18nKey.BUTTON$SHOW_AGENT_TOOLS_AND_METADATA)}
          />
        </ContextMenuListItem>
      )}

      {onShowMicroagents && !isV1Conversation && (
        <ContextMenuListItem
          testId="show-microagents-button"
          onClick={onShowMicroagents}
        >
          <ContextMenuIconText
            icon={Bot}
            text={t(I18nKey.CONVERSATION$SHOW_MICROAGENTS)}
          />
        </ContextMenuListItem>
      )}

      {hasTools && (hasInfo || hasControl) && <Divider />}

      {onDisplayCost && (
        <ContextMenuListItem
          testId="display-cost-button"
          onClick={onDisplayCost}
        >
          <ContextMenuIconText
            icon={Wallet}
            text={t(I18nKey.BUTTON$DISPLAY_COST)}
          />
        </ContextMenuListItem>
      )}

      {hasInfo && hasControl && <Divider />}

      {onStop && (
        <ContextMenuListItem testId="stop-button" onClick={onStop}>
          <ContextMenuIconText icon={Power} text={t(I18nKey.BUTTON$PAUSE)} />
        </ContextMenuListItem>
      )}

      {onDelete && (
        <ContextMenuListItem testId="delete-button" onClick={onDelete}>
          <ContextMenuIconText icon={Trash} text={t(I18nKey.BUTTON$DELETE)} />
        </ContextMenuListItem>
      )}
    </ContextMenu>
  );
}
