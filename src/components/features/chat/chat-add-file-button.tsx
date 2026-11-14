import React from "react";
import { Paperclip, Plus } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { chatInputIconButtonClassName } from "#/utils/form-control-classes";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useCloseOnEscape } from "#/hooks/use-close-on-escape";
import { useConversationNameContextMenu } from "#/hooks/use-conversation-name-context-menu";
import { ToolsContextMenu } from "#/components/features/controls/tools-context-menu";
import { SystemMessageModal } from "#/components/features/conversation-panel/system-message-modal";
import { SkillsModal } from "#/components/features/conversation-panel/skills-modal";
import { PluginsModal } from "#/components/features/conversation-panel/plugins-modal";
import { HooksModal } from "#/components/features/conversation-panel/hooks-modal";
import {
  useConversationConfigurationCapability,
  useConversationMcpCapabilities,
} from "#/hooks/query/use-conversation-configuration-capability";

const ConversationConfigurationModal = React.lazy(() =>
  import("./conversation-configuration-modal").then((module) => ({
    default: module.ConversationConfigurationModal,
  })),
);

export interface ChatAddFileButtonProps {
  handleFileIconClick: () => void;
  disabled?: boolean;
  /**
   * Offer the "Switch agent profile" submenu. Computed by ChatInputActions:
   * only while starting a new conversation (home or a blank conversation) and
   * only when the backend has agent profiles (OSS-5735 — once a conversation
   * starts, the agent profile is locked).
   */
  showAgentProfileSwitch?: boolean;
}

export function ChatAddFileButton({
  handleFileIconClick,
  disabled = false,
  showAgentProfileSwitch = false,
}: ChatAddFileButtonProps) {
  const { t } = useTranslation("openhands");
  const { conversationId } = useOptionalConversationId();
  const { data: conversation } = useActiveConversation();
  const canConfigureConversation =
    !!conversationId && conversation?.id === conversationId && !disabled;
  const { data: canConfigure } = useConversationConfigurationCapability(
    canConfigureConversation,
  );
  const { canInstall } = useConversationMcpCapabilities(
    canConfigureConversation,
  );
  const [configurationOpen, setConfigurationOpen] = React.useState(false);
  const [menuOpen, setMenuOpen] = React.useState(false);
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  useCloseOnEscape(menuOpen, () => setMenuOpen(false), triggerRef);

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
    onContextMenuToggle: setMenuOpen,
  });

  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (disabled) return;
    setMenuOpen((open) => !open);
  };

  return (
    <div className="relative">
      <button
        ref={triggerRef}
        type="button"
        className={cn(
          chatInputIconButtonClassName,
          "relative shrink-0 size-6",
          disabled ? "cursor-not-allowed text-text-subtle" : undefined,
          menuOpen && !disabled && "text-contrast bg-contrast/10",
        )}
        aria-label={t(I18nKey.CHAT_INTERFACE$PLUS_MENU)}
        aria-expanded={menuOpen}
        aria-haspopup="menu"
        data-testid="chat-plus-button"
        onClick={handleClick}
        disabled={disabled}
      >
        <span className="flex h-full w-full items-center justify-center">
          <Plus className="h-3.25 w-3.25 shrink-0" strokeWidth={2} />
        </span>
      </button>

      {menuOpen && (
        <ToolsContextMenu
          onClose={() => setMenuOpen(false)}
          showAgentProfileSwitch={showAgentProfileSwitch}
          onShowSkills={handleShowSkills}
          onShowPlugins={handleShowPlugins}
          onShowHooks={handleShowHooks}
          onShowAgentTools={handleShowAgentTools}
          shouldShowAgentTools={shouldShowAgentTools}
          shouldShowHooks={shouldShowHooks}
          shouldShowPlugins={shouldShowPlugins}
          configurationAction={
            canConfigure && !disabled
              ? {
                  label: t(I18nKey.CHAT$CONVERSATION_CONFIGURATION),
                  onClick: () => setConfigurationOpen(true),
                }
              : undefined
          }
          footerAction={{
            testId: "add-files-and-images-button",
            icon: (
              <Paperclip
                className="h-4 w-4 shrink-0"
                strokeWidth={2}
                aria-hidden
              />
            ),
            label: t(I18nKey.CHAT_INTERFACE$ADD_FILES_AND_IMAGES),
            onClick: handleFileIconClick,
          }}
        />
      )}

      {configurationOpen &&
        canConfigure &&
        conversationId === conversation?.id && (
          <React.Suspense fallback={null}>
            <ConversationConfigurationModal
              key={conversationId}
              conversationId={conversationId}
              isAcp={conversation?.agent_kind === "acp"}
              canInstallMcp={canInstall}
              onClose={() => setConfigurationOpen(false)}
            />
          </React.Suspense>
        )}

      <SystemMessageModal
        isOpen={systemModalVisible}
        onClose={() => setSystemModalVisible(false)}
        systemMessage={systemMessage || null}
      />
      {skillsModalVisible && (
        <SkillsModal onClose={() => setSkillsModalVisible(false)} />
      )}
      {pluginsModalVisible && (
        <PluginsModal onClose={() => setPluginsModalVisible(false)} />
      )}
      {hooksModalVisible && (
        <HooksModal onClose={() => setHooksModalVisible(false)} />
      )}
    </div>
  );
}
