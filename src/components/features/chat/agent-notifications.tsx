import { useTranslation } from "react-i18next";
import XMarkIcon from "#/icons/x-mark.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { AgentNotificationsList } from "./agent-notifications-list";
import { AgentNotificationsTitle } from "./agent-notifications-title";
import type { AgentNotification } from "./agent-notifications.constants";

const DISMISS_BUTTON_CLASSNAME = cn(
  "inline-flex size-6 shrink-0 cursor-pointer items-center justify-center rounded-sm",
  "text-muted transition-colors hover:bg-white/10 hover:text-white",
);

interface AgentNotificationsProps {
  agentNotifications: AgentNotification[];
  onCreateAll: (selectedIds: string[]) => void;
  onDismiss: () => void;
  onRemove: (id: string) => void;
  disabled?: boolean;
  isCreating?: boolean;
}

export function AgentNotifications({
  agentNotifications,
  onCreateAll,
  onDismiss,
  onRemove,
  disabled = false,
  isCreating = false,
}: AgentNotificationsProps) {
  const { t } = useTranslation("openhands");
  const controlsDisabled = disabled || isCreating;

  if (agentNotifications.length === 0) {
    return null;
  }

  return (
    <section
      aria-label={t(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_TITLE)}
      data-testid="agent-notifications"
      className={cn(
        "mb-2 rounded-lg border border-[var(--oh-border)]",
        "bg-[var(--oh-surface-raised)] px-3 py-3",
      )}
    >
      <div className="mb-3 flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <AgentNotificationsTitle />
        </div>
        <button
          type="button"
          data-testid="agent-notifications-dismiss"
          aria-label={t(I18nKey.BUTTON$CLOSE)}
          onClick={onDismiss}
          disabled={controlsDisabled}
          className={cn(
            DISMISS_BUTTON_CLASSNAME,
            controlsDisabled && "cursor-not-allowed opacity-50",
          )}
        >
          <XMarkIcon className="size-3.5" aria-hidden />
        </button>
      </div>

      <AgentNotificationsList
        agentNotifications={agentNotifications}
        onSubmit={onCreateAll}
        onRemove={onRemove}
        onDismiss={onDismiss}
        disabled={disabled}
        isSubmitting={isCreating}
        submitTestId="agent-notifications-create-all"
        dismissTestId="agent-notifications-dismiss-action"
        listItemTestIdPrefix="agent-notification"
      />
    </section>
  );
}
