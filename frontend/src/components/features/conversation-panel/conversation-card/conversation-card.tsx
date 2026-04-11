import React from "react";
import { usePostHog } from "posthog-js/react";
import { cn } from "#/utils/utils";
import { transformVSCodeUrl } from "#/utils/vscode-url-helper";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { V1SandboxStatus } from "#/api/sandbox-service/sandbox-service.types";
import { RepositorySelection } from "#/api/open-hands.types";
import { ConversationCardHeader } from "./conversation-card-header";
import { ConversationCardActions } from "./conversation-card-actions";
import { ConversationCardFooter } from "./conversation-card-footer";
import { SandboxStatusBadges } from "./sandbox-status-badges";
import { useDownloadConversation } from "#/hooks/use-download-conversation";

interface ConversationCardProps {
  onClick?: () => void;
  onDelete?: () => void;
  onStop?: () => void;
  onChangeTitle?: (title: string) => void;
  showOptions?: boolean;
  title: string;
  selectedRepository: RepositorySelection | null;
  lastUpdatedAt: string; // ISO 8601
  createdAt?: string; // ISO 8601
  sandboxStatus?: V1SandboxStatus;
  conversationId?: string; // Optional conversation ID for VS Code URL
  contextMenuOpen?: boolean;
  onContextMenuToggle?: (isOpen: boolean) => void;
  llmModel?: string | null;
}

export function ConversationCard({
  onClick,
  onDelete,
  onStop,
  onChangeTitle,
  showOptions,
  title,
  selectedRepository,
  // lastUpdatedAt is kept in props for backward compatibility
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  lastUpdatedAt,
  createdAt,
  conversationId,
  sandboxStatus,
  contextMenuOpen = false,
  onContextMenuToggle,
  llmModel,
}: ConversationCardProps) {
  const posthog = usePostHog();
  const [titleMode, setTitleMode] = React.useState<"view" | "edit">("view");
  const { mutateAsync: downloadConversation } = useDownloadConversation();

  const onTitleSave = (newTitle: string) => {
    if (newTitle !== "" && newTitle !== title) {
      onChangeTitle?.(newTitle);
    }
    setTitleMode("view");
  };

  const handleDelete = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    onDelete?.();
    onContextMenuToggle?.(false);
  };

  const handleStop = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    onStop?.();
    onContextMenuToggle?.(false);
  };

  const handleEdit = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setTitleMode("edit");
    onContextMenuToggle?.(false);
  };

  const handleDownloadViaVSCode = async (
    event: React.MouseEvent<HTMLButtonElement>,
  ) => {
    event.preventDefault();
    event.stopPropagation();
    posthog.capture("download_via_vscode_button_clicked");

    // Fetch the VS Code URL from the API
    if (conversationId) {
      try {
        const data = await ConversationService.getVSCodeUrl(conversationId);
        if (data.vscode_url) {
          const transformedUrl = transformVSCodeUrl(data.vscode_url);
          if (transformedUrl) {
            window.open(transformedUrl, "_blank");
          }
        }
        // VS Code URL not available
      } catch {
        // Failed to fetch VS Code URL
      }
    }

    onContextMenuToggle?.(false);
  };

  const handleDownloadConversation = async (
    event: React.MouseEvent<HTMLButtonElement>,
  ) => {
    event.preventDefault();
    event.stopPropagation();

    if (conversationId) {
      await downloadConversation(conversationId);
    }
    onContextMenuToggle?.(false);
  };

  const hasContextMenu = !!(onDelete || onChangeTitle || showOptions);

  return (
    <div
      data-testid="conversation-card"
      data-context-menu-open={contextMenuOpen.toString()}
      onClick={onClick}
      className={cn(
        "relative h-auto w-full p-3.5 border-b border-neutral-600 cursor-pointer",
        "data-[context-menu-open=false]:hover:bg-[#454545]",
      )}
    >
      <div className="flex items-center justify-between w-full">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <ConversationCardHeader
            title={title}
            titleMode={titleMode}
            onTitleSave={onTitleSave}
            sandboxStatus={sandboxStatus}
          />
          <SandboxStatusBadges sandboxStatus={sandboxStatus} />
        </div>

        {hasContextMenu && (
          <ConversationCardActions
            contextMenuOpen={contextMenuOpen}
            onContextMenuToggle={onContextMenuToggle || (() => {})}
            onDelete={onDelete && handleDelete}
            onStop={onStop && handleStop}
            onEdit={onChangeTitle && handleEdit}
            onDownloadViaVSCode={handleDownloadViaVSCode}
            onDownloadConversation={handleDownloadConversation}
            sandboxStatus={sandboxStatus}
            conversationId={conversationId}
            showOptions={showOptions}
          />
        )}
      </div>

      <ConversationCardFooter
        selectedRepository={selectedRepository}
        lastUpdatedAt={lastUpdatedAt}
        createdAt={createdAt}
        sandboxStatus={sandboxStatus}
        llmModel={llmModel}
      />
    </div>
  );
}
