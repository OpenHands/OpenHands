import React from "react";
import { useParams } from "react-router";
import { useTranslation } from "react-i18next";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useUpdateConversation } from "#/hooks/mutation/use-update-conversation";
import { useConversationNameContextMenu } from "#/hooks/use-conversation-name-context-menu";
import { displaySuccessToast } from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import { EllipsisButton } from "../conversation-panel/ellipsis-button";
import { ConversationNameContextMenu } from "./conversation-name-context-menu";
import { SystemMessageModal } from "../conversation-panel/system-message-modal";
import { SkillsModal } from "../conversation-panel/skills-modal";
import { ConfirmDeleteModal } from "../conversation-panel/confirm-delete-modal";
import { ConfirmStopModal } from "../conversation-panel/confirm-stop-modal";
import { MetricsModal } from "./metrics-modal/metrics-modal";
import { ConversationVersionBadge } from "../conversation-panel/conversation-card/conversation-version-badge";

export function ConversationName() {
  const { t } = useTranslation();
  const { conversationId } = useParams<{ conversationId: string }>();
  const { data: conversation } = useActiveConversation();
  const { mutate: updateConversation } = useUpdateConversation();

  const [titleMode, setTitleMode] = React.useState<"view" | "edit">("view");
  const [contextMenuOpen, setContextMenuOpen] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);

  // Use the custom hook for context menu handlers
  const {
    handleDelete,
    handleStop,
    handleDownloadViaVSCode,
    handleDownloadConversation,
    handleDisplayCost,
    handleShowAgentTools,
    handleShowSkills,
    handleExportConversation,
    handleTogglePublic,
    handleCopyShareLink,
    shareUrl,
    handleConfirmDelete,
    handleConfirmStop,
    metricsModalVisible,
    setMetricsModalVisible,
    systemModalVisible,
    setSystemModalVisible,
    skillsModalVisible,
    setSkillsModalVisible,
    confirmDeleteModalVisible,
    setConfirmDeleteModalVisible,
    confirmStopModalVisible,
    setConfirmStopModalVisible,
    systemMessage,
    shouldShowStop,
    shouldShowDownload,
    shouldShowExport,
    shouldShowDownloadConversation,
    shouldShowDisplayCost,
    shouldShowAgentTools,
    shouldShowSkills,
  } = useConversationNameContextMenu({
    conversationId,
    conversationStatus: conversation?.status,
    showOptions: true, // Enable all options for conversation name
    onContextMenuToggle: setContextMenuOpen,
  });

  const handleDoubleClick = () => {
    setTitleMode("edit");
  };

  const handleBlur = () => {
    if (inputRef.current?.value && conversationId) {
      const trimmed = inputRef.current.value.trim();
      if (trimmed !== conversation?.title) {
        updateConversation(
          { conversationId, newTitle: trimmed },
          {
            onSuccess: () => {
              displaySuccessToast(t(I18nKey.CONVERSATION$TITLE_UPDATED));
            },
          },
        );
      }
    } else if (inputRef.current) {
      // reset the value if it's empty
      inputRef.current.value = conversation?.title ?? "";
    }

    setTitleMode("view");
  };

  const handleKeyUp = (event: React.KeyboardEvent<HTMLInputElement>) => {
    // Ignore Enter key during IME composition (e.g., Chinese, Japanese, Korean input)
    if (event.nativeEvent.isComposing) {
      return;
    }
    if (event.key === "Enter") {
      event.currentTarget.blur();
    }
  };

  const handleInputClick = (event: React.MouseEvent<HTMLInputElement>) => {
    if (titleMode === "edit") {
      event.preventDefault();
      event.stopPropagation();
    }
  };

  const handleEllipsisClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setContextMenuOpen(!contextMenuOpen);
  };

  const handleRename = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setTitleMode("edit");
    setContextMenuOpen(false);
  };

  React.useEffect(() => {
    if (titleMode === "edit") {
      inputRef.current?.focus();
    }
  }, [titleMode]);

  if (!conversation) {
    return null;
  }

  return (
    <>
      <div
        className="flex items-center gap-2 h-[22px] text-base font-normal text-left pl-0 lg:pl-1"
        data-testid="conversation-name"
      >
        {titleMode === "edit" ? (
          <input
            ref={inputRef}
            data-testid="conversation-name-input"
            onClick={handleInputClick}
            onBlur={handleBlur}
            onKeyUp={handleKeyUp}
            type="text"
            defaultValue={conversation.title}
            className="text-white leading-5 bg-transparent border-none outline-none text-base font-normal w-fit max-w-fit field-sizing-content"
          />
        ) : (
          <div
            className="text-white leading-5 w-fit max-w-fit truncate"
            data-testid="conversation-name-title"
            onDoubleClick={handleDoubleClick}
            title={conversation.title}
          >
            {conversation.title}
          </div>
        )}

        {titleMode !== "edit" && (
          <ConversationVersionBadge
            version={conversation.conversation_version}
          />
        )}

        {titleMode !== "edit" && (
          <div className="relative flex items-center">
            <EllipsisButton fill="#B1B9D3" onClick={handleEllipsisClick} />
            {contextMenuOpen && (
              <ConversationNameContextMenu
                onClose={() => setContextMenuOpen(false)}
                onRename={handleRename}
                onDelete={handleDelete}
                onStop={shouldShowStop ? handleStop : undefined}
                onDisplayCost={
                  shouldShowDisplayCost ? handleDisplayCost : undefined
                }
                onShowAgentTools={
                  shouldShowAgentTools ? handleShowAgentTools : undefined
                }
                onShowSkills={shouldShowSkills ? handleShowSkills : undefined}
                onExportConversation={
                  shouldShowExport ? handleExportConversation : undefined
                }
                onDownloadViaVSCode={
                  shouldShowDownload ? handleDownloadViaVSCode : undefined
                }
                onTogglePublic={handleTogglePublic}
                shareUrl={shareUrl}
                onCopyShareLink={handleCopyShareLink}
                onDownloadConversation={
                  shouldShowDownloadConversation
                    ? handleDownloadConversation
                    : undefined
                }
                position="bottom"
              />
            )}
          </div>
        )}
      </div>

      {/* Metrics Modal */}
      <MetricsModal
        isOpen={metricsModalVisible}
        onOpenChange={setMetricsModalVisible}
      />

      {/* System Message Modal */}
      <SystemMessageModal
        isOpen={systemModalVisible}
        onClose={() => setSystemModalVisible(false)}
        systemMessage={systemMessage || null}
      />

      {/* Skills Modal */}
      {skillsModalVisible && (
        <SkillsModal onClose={() => setSkillsModalVisible(false)} />
      )}

      {/* Confirm Delete Modal */}
      {confirmDeleteModalVisible && (
        <ConfirmDeleteModal
          onConfirm={handleConfirmDelete}
          onCancel={() => setConfirmDeleteModalVisible(false)}
          conversationTitle={conversation?.title}
        />
      )}

      {/* Confirm Stop Modal */}
      {confirmStopModalVisible && (
        <ConfirmStopModal
          onConfirm={handleConfirmStop}
          onCancel={() => setConfirmStopModalVisible(false)}
        />
      )}
    </>
  );
}
