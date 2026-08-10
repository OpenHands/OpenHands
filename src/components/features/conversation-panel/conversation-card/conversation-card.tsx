import React from "react";
import { Pin } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useTracking } from "#/hooks/use-tracking";
import { cn } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";
import { transformVSCodeUrl } from "#/utils/vscode-url-helper";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { getDisplayConversationTags } from "#/api/agent-server-adapter";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import { SandboxStatus } from "#/api/conversation-service/agent-server-conversation-service.types";
import { RepositorySelection } from "#/api/open-hands.types";
import { formatTimeDelta } from "#/utils/format-time-delta";
import { ConversationCardHeader } from "./conversation-card-header";
import { ConversationCardActions } from "./conversation-card-actions";
import { ConversationCardFooter } from "./conversation-card-footer";
import { ConversationStatusBadges } from "./conversation-status-badges";
import { useDownloadConversation } from "#/hooks/use-download-conversation";

interface ConversationCardProps {
  onClick?: () => void;
  onDelete?: () => void;
  onStop?: () => void;
  onChangeTitle?: (title: string) => void;
  showOptions?: boolean;
  title: string;
  selectedRepository: RepositorySelection | null;
  lastUpdatedAt: string;
  createdAt?: string;
  executionStatus?: ExecutionStatus | null;
  sandboxStatus?: SandboxStatus | null;
  conversationId?: string;
  contextMenuOpen?: boolean;
  onContextMenuToggle?: (isOpen: boolean) => void;
  isActive?: boolean;
  workspaceWorkingDir?: string | null;
  showRepositoryMetadata?: boolean;
  llmModel?: string | null;
  showLlmProfiles?: boolean;
  agentKind?: "openhands" | "acp" | null;
  acpServer?: string | null;
  tags?: Record<string, string> | null;
  /** Gates the tag-chip row; wired to the panel's "Tags" metadata toggle. */
  showTags?: boolean;
  isPinned?: boolean;
  onTogglePin?: () => void;
  /** When true and pinned, keep the pin icon visible without hovering. */
  alwaysShowPinIcon?: boolean;
}

export function ConversationCard({
  onClick,
  onDelete,
  onStop,
  onChangeTitle,
  showOptions,
  title,
  selectedRepository,
  lastUpdatedAt,
  createdAt,
  conversationId,
  executionStatus,
  sandboxStatus,
  contextMenuOpen = false,
  onContextMenuToggle,
  isActive = false,
  workspaceWorkingDir,
  showRepositoryMetadata = true,
  llmModel = null,
  showLlmProfiles = false,
  agentKind = null,
  acpServer = null,
  tags = null,
  showTags = false,
  isPinned = false,
  onTogglePin,
  alwaysShowPinIcon = false,
}: ConversationCardProps) {
  const { t } = useTranslation("openhands");
  const { trackDownloadVsCodeButtonClicked } = useTracking();
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
    trackDownloadVsCodeButtonClicked();

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

  const handleTogglePin = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    onTogglePin?.();
  };

  const renderPinButton = () => (
    <button
      type="button"
      data-testid={
        conversationId
          ? `conversation-pin-toggle-${conversationId}`
          : "conversation-pin-toggle"
      }
      aria-pressed={isPinned}
      aria-label={
        isPinned
          ? t(I18nKey.CONVERSATION_PANEL$UNPIN_CONVERSATION)
          : t(I18nKey.CONVERSATION_PANEL$PIN_CONVERSATION)
      }
      onClick={handleTogglePin}
      className={cn(
        "flex shrink-0 cursor-pointer items-center justify-center rounded-md p-1",
        "text-[var(--oh-muted)] hover:bg-white/10 hover:text-white",
      )}
    >
      <Pin
        className={cn("h-3.5 w-3.5", isPinned && "fill-current")}
        aria-hidden
      />
    </button>
  );

  const hasContextMenu = !!(onDelete || onChangeTitle || showOptions);
  const hasHoverActions = hasContextMenu || !!onTogglePin;
  const showPersistentPinIcon = alwaysShowPinIcon && isPinned && !!onTogglePin;
  const shouldRenderFooter =
    showRepositoryMetadata ||
    (showLlmProfiles && (agentKind === "acp" || !!llmModel)) ||
    (showTags && getDisplayConversationTags(tags).length > 0);

  return (
    <div
      data-testid="conversation-card"
      data-context-menu-open={contextMenuOpen.toString()}
      data-active={isActive ? "true" : "false"}
      onClick={onClick}
      className={cn(
        "group relative h-auto w-full cursor-pointer rounded-md py-1 pl-2 pr-1 transition-colors",
        !contextMenuOpen && "hover:bg-[var(--oh-surface)]",
        (isActive || contextMenuOpen) && "bg-[var(--oh-surface)]",
      )}
    >
      <div className="flex items-center w-full min-w-0">
        <div className="flex items-center gap-2 flex-1 min-w-0 overflow-hidden">
          <ConversationCardHeader
            title={title}
            titleMode={titleMode}
            onTitleSave={onTitleSave}
            executionStatus={executionStatus}
            sandboxStatus={sandboxStatus}
          />
          {sandboxStatus === "ERROR" && <ConversationStatusBadges />}
        </div>

        <div
          className={cn(
            "relative ml-auto pl-2 flex items-center justify-end shrink-0",
            // The hover action overlay (pin + ellipsis) is absolutely
            // positioned, so reserve its width so the flex-1 title truncates
            // instead of colliding with the buttons. Pinned cards keep the pin
            // visible at rest, so reserve the width always for those.
            showPersistentPinIcon
              ? "min-w-[3.75rem]"
              : hasHoverActions &&
                  "group-hover:min-w-[3.75rem] group-focus-within:min-w-[3.75rem]",
            contextMenuOpen && "min-w-[3.75rem]",
          )}
        >
          {!showPersistentPinIcon && (createdAt ?? lastUpdatedAt) && (
            <p
              className={cn(
                "text-xs text-[var(--oh-muted)] text-right whitespace-nowrap transition-opacity -translate-x-1.5",
                hasHoverActions &&
                  "group-hover:opacity-0 group-focus-within:opacity-0",
                contextMenuOpen && "opacity-0",
              )}
            >
              <time>{formatTimeDelta(lastUpdatedAt ?? createdAt)}</time>
            </p>
          )}

          {hasHoverActions ? (
            <div
              className={cn(
                "absolute right-0 top-1/2 flex -translate-y-1/2 items-center gap-0.5 transition-opacity",
                showPersistentPinIcon
                  ? "pointer-events-auto visible opacity-100"
                  : cn(
                      "pointer-events-none opacity-0 invisible",
                      "group-hover:pointer-events-auto group-hover:opacity-100 group-hover:visible",
                      "group-focus-within:pointer-events-auto group-focus-within:opacity-100 group-focus-within:visible",
                    ),
                contextMenuOpen && "pointer-events-auto visible opacity-100",
              )}
            >
              {onTogglePin ? renderPinButton() : null}
              {showPersistentPinIcon &&
              (createdAt ?? lastUpdatedAt) &&
              hasContextMenu ? (
                <div className="relative shrink-0">
                  <div
                    className={cn(
                      !contextMenuOpen &&
                        "invisible pointer-events-none group-hover:visible group-hover:pointer-events-auto group-focus-within:visible group-focus-within:pointer-events-auto",
                    )}
                  >
                    <ConversationCardActions
                      contextMenuOpen={contextMenuOpen}
                      onContextMenuToggle={onContextMenuToggle || (() => {})}
                      onDelete={onDelete && handleDelete}
                      onStop={onStop && handleStop}
                      onEdit={onChangeTitle && handleEdit}
                      onDownloadViaVSCode={handleDownloadViaVSCode}
                      onDownloadConversation={handleDownloadConversation}
                      executionStatus={executionStatus}
                      conversationId={conversationId}
                      showOptions={showOptions}
                    />
                  </div>
                  <p
                    className={cn(
                      "pointer-events-none absolute inset-0 flex items-center justify-end",
                      "text-xs text-[var(--oh-muted)] whitespace-nowrap -translate-x-1.5",
                      "group-hover:hidden group-focus-within:hidden",
                      contextMenuOpen && "hidden",
                    )}
                  >
                    <time>{formatTimeDelta(lastUpdatedAt ?? createdAt)}</time>
                  </p>
                </div>
              ) : null}
              {!showPersistentPinIcon && hasContextMenu ? (
                <ConversationCardActions
                  contextMenuOpen={contextMenuOpen}
                  onContextMenuToggle={onContextMenuToggle || (() => {})}
                  onDelete={onDelete && handleDelete}
                  onStop={onStop && handleStop}
                  onEdit={onChangeTitle && handleEdit}
                  onDownloadViaVSCode={handleDownloadViaVSCode}
                  onDownloadConversation={handleDownloadConversation}
                  executionStatus={executionStatus}
                  conversationId={conversationId}
                  showOptions={showOptions}
                />
              ) : null}
            </div>
          ) : null}
        </div>
      </div>

      {shouldRenderFooter && (
        <ConversationCardFooter
          selectedRepository={selectedRepository}
          lastUpdatedAt={lastUpdatedAt}
          createdAt={createdAt}
          executionStatus={executionStatus}
          workspaceWorkingDir={workspaceWorkingDir}
          showRepositoryMetadata={showRepositoryMetadata}
          showTimestamp={false}
          llmModel={llmModel}
          showAgentChip={showLlmProfiles}
          agentKind={agentKind}
          acpServer={acpServer}
          tags={tags}
          showTags={showTags}
        />
      )}
    </div>
  );
}
