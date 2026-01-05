import React from "react";
import { cn } from "#/utils/utils";
import { ConversationStatus } from "#/types/conversation-status";
import { ConversationCardContextMenu } from "./conversation-card-context-menu";
import EllipsisIcon from "#/icons/ellipsis.svg?react";

interface ConversationCardActionsProps {
  contextMenuOpen: boolean;
  onContextMenuToggle: (isOpen: boolean) => void;
  onDelete?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onStop?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onEdit?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onDownloadViaVSCode?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onDownloadConversation?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  conversationStatus?: ConversationStatus;
  conversationId?: string;
  showOptions?: boolean;
}

export function ConversationCardActions({
  contextMenuOpen,
  onContextMenuToggle,
  onDelete,
  onStop,
  onEdit,
  onDownloadViaVSCode,
  onDownloadConversation,
  conversationStatus,
  conversationId,
  showOptions,
}: ConversationCardActionsProps) {
  const isConversationArchived = conversationStatus === "ARCHIVED";

  return (
    <div className="group">
      <button
        data-testid="ellipsis-button"
        type="button"
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          onContextMenuToggle(!contextMenuOpen);
        }}
        className={cn(
          "cursor-pointer w-6 h-6 flex flex-row items-center justify-center translate-x-2.5",
          isConversationArchived && "opacity-60",
        )}
      >
        <EllipsisIcon />
      </button>
      <div
        className={cn(
          // Show on hover (desktop) or when explicitly opened (click/touch)
          "relative opacity-0 invisible group-hover:opacity-100 group-hover:visible",
          // Override hover styles when explicitly opened via click
          contextMenuOpen && "opacity-100 visible",
        )}
      >
        <ConversationCardContextMenu
          onClose={() => onContextMenuToggle(false)}
          onDelete={onDelete}
          onStop={
            conversationStatus === "RUNNING" ||
            conversationStatus === "STARTING"
              ? onStop
              : undefined
          }
          onEdit={onEdit}
          onDownloadViaVSCode={
            conversationId && showOptions ? onDownloadViaVSCode : undefined
          }
          onDownloadConversation={
            conversationId ? onDownloadConversation : undefined
          }
          position="bottom"
        />
      </div>
    </div>
  );
}
