import React from "react";
import { Wrench } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { ComboboxCaretInline } from "#/ui/combobox-caret";
import { chatInputPillButtonClassName } from "#/utils/form-control-classes";
import { ToolsContextMenu } from "./tools-context-menu";
import { useConversationNameContextMenu } from "#/hooks/use-conversation-name-context-menu";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { SystemMessageModal } from "../conversation-panel/system-message-modal";
import { SkillsModal } from "../conversation-panel/skills-modal";
import { PluginsModal } from "../conversation-panel/plugins-modal";
import { HooksModal } from "../conversation-panel/hooks-modal";
import { cn } from "#/utils/utils";

export function Tools() {
  const { t } = useTranslation("openhands");
  // Optional because this control also renders inside the home-page chat
  // input shell, before any conversation exists.
  const { conversationId } = useOptionalConversationId();
  const { data: conversation } = useActiveConversation();
  const [contextMenuOpen, setContextMenuOpen] = React.useState(false);

  const {
    handleShowAgentTools,
    handleShowSkills,
    handleShowPlugins,
    handleShowHooks,
    systemModalVisible,
    setSystemModalVisible,
    skillsModalVisible,
    setSkillsModalVisible,
    pluginsModalVisible,
    setPluginsModalVisible,
    hooksModalVisible,
    setHooksModalVisible,
    systemMessage,
    shouldShowAgentTools,
    shouldShowHooks,
    shouldShowPlugins,
  } = useConversationNameContextMenu({
    conversationId: conversationId ?? undefined,
    executionStatus: conversation?.execution_status,
    showOptions: true,
    onContextMenuToggle: setContextMenuOpen,
  });

  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setContextMenuOpen(!contextMenuOpen);
  };

  return (
    <div className="relative">
      <button
        type="button"
        className={cn(chatInputPillButtonClassName, "group flex")}
        onClick={handleClick}
      >
        <Wrench
          className="h-[13px] w-[13px] shrink-0"
          strokeWidth={2}
          aria-hidden="true"
        />
        <span className="text-sm font-normal leading-5">
          {t(I18nKey.MICROAGENTS_MODAL$TOOLS)}
        </span>
        <ComboboxCaretInline isOpen={contextMenuOpen} />
      </button>
      {contextMenuOpen && (
        <ToolsContextMenu
          onClose={() => setContextMenuOpen(false)}
          onShowSkills={handleShowSkills}
          onShowPlugins={handleShowPlugins}
          onShowHooks={handleShowHooks}
          onShowAgentTools={handleShowAgentTools}
          shouldShowAgentTools={shouldShowAgentTools}
          shouldShowHooks={shouldShowHooks}
          shouldShowPlugins={shouldShowPlugins}
        />
      )}

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

      {/* Plugins Modal */}
      {pluginsModalVisible && (
        <PluginsModal onClose={() => setPluginsModalVisible(false)} />
      )}

      {/* Hooks Modal */}
      {hooksModalVisible && (
        <HooksModal onClose={() => setHooksModalVisible(false)} />
      )}
    </div>
  );
}
