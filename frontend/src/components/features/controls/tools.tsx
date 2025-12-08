import React from "react";
import { useParams } from "react-router";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import ToolsIcon from "#/icons/tools.svg?react";
import { ToolsContextMenu } from "./tools-context-menu";
import { useConversationNameContextMenu } from "#/hooks/use-conversation-name-context-menu";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { SystemMessageModal } from "../conversation-panel/system-message-modal";
import { MicroagentsModal } from "../conversation-panel/microagents-modal";

export function Tools() {
  const { t } = useTranslation();
  const { conversationId } = useParams<{ conversationId: string }>();
  const { data: conversation } = useActiveConversation();
  const [contextMenuOpen, setContextMenuOpen] = React.useState(false);

  const {
    handleShowAgentTools,
    handleShowMicroagents,
    systemModalVisible,
    setSystemModalVisible,
    microagentsModalVisible,
    setMicroagentsModalVisible,
    systemMessage,
    shouldShowAgentTools,
  } = useConversationNameContextMenu({
    conversationId,
    conversationStatus: conversation?.status,
    showOptions: true, // Enable all options for conversation name
    onContextMenuToggle: setContextMenuOpen,
  });

  const handleClick = (event: React.MouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setContextMenuOpen(!contextMenuOpen);
  };

  return (
    <div className="relative">
      <div
        className="flex items-center gap-1 cursor-pointer"
        onClick={handleClick}
      >
        <ToolsIcon width={18} height={18} color="#959CB2" />
        <span className="text-sm font-normal leading-5 text-white">
          {t(I18nKey.MICROAGENTS_MODAL$TOOLS)}
        </span>
      </div>
      {contextMenuOpen && (
        <ToolsContextMenu
          onClose={() => setContextMenuOpen(false)}
          onShowMicroagents={handleShowMicroagents}
          onShowAgentTools={handleShowAgentTools}
          shouldShowAgentTools={shouldShowAgentTools}
        />
      )}

      {/* System Message Modal */}
      <SystemMessageModal
        isOpen={systemModalVisible}
        onClose={() => setSystemModalVisible(false)}
        systemMessage={systemMessage ? systemMessage.args : null}
      />

      {/* Microagents Modal */}
      {microagentsModalVisible && (
        <MicroagentsModal onClose={() => setMicroagentsModalVisible(false)} />
      )}
    </div>
  );
}
