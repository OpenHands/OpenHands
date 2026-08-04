import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { AgentNotificationsList } from "#/components/features/chat/agent-notifications-list";
import { AgentNotificationsTitle } from "#/components/features/chat/agent-notifications-title";
import type { AgentNotification } from "#/components/features/chat/agent-notifications.constants";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { modalTitleLgClassName } from "#/utils/modal-classes";

interface SidebarOnboardingAgentNotificationsModalProps {
  agentNotifications: AgentNotification[];
  isOpen: boolean;
  isCreating?: boolean;
  onClose: () => void;
  onCreateAll: (selectedIds: string[]) => void;
}

export function SidebarOnboardingAgentNotificationsModal({
  agentNotifications,
  isOpen,
  isCreating = false,
  onClose,
  onCreateAll,
}: SidebarOnboardingAgentNotificationsModalProps) {
  const { t } = useTranslation("openhands");

  if (!isOpen || agentNotifications.length === 0) {
    return null;
  }

  const handleCreateAll = (selectedIds: string[]) => {
    onCreateAll(selectedIds);
    onClose();
  };

  return (
    <ModalBackdrop
      onClose={isCreating ? undefined : onClose}
      closeOnEscape={!isCreating}
      closeOnBackdropClick={!isCreating}
      aria-label={t(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_TITLE)}
    >
      <div
        data-testid="sidebar-onboarding-agent-notifications-modal"
        className={cn(
          "relative flex w-[480px] max-w-[90vw] flex-col",
          "rounded-xl border border-[var(--oh-border)] bg-base-secondary",
        )}
      >
        <ModalCloseButton
          onClose={onClose}
          testId="sidebar-onboarding-agent-notifications-modal-close"
          disabled={isCreating}
        />

        <header className="flex-shrink-0 px-6 pb-2 pt-6">
          <AgentNotificationsTitle
            as="h2"
            titleClassName={cn("pr-8", modalTitleLgClassName)}
            infoTestId="sidebar-onboarding-agent-notifications-info"
          />
        </header>

        <div className="flex-shrink-0 px-4 pb-6">
          <AgentNotificationsList
            agentNotifications={agentNotifications}
            onSubmit={handleCreateAll}
            isSubmitting={isCreating}
            submitTestId="sidebar-onboarding-agent-notifications-create-all"
            listItemTestIdPrefix="sidebar-onboarding-agent-notification"
          />
        </div>
      </div>
    </ModalBackdrop>
  );
}
