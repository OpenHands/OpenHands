import { Sparkles } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

interface AgentNotificationsDropdownEmptyStateProps {
  testId?: string;
}

export function AgentNotificationsDropdownEmptyState({
  testId = "agent-notifications-bell-empty-state",
}: AgentNotificationsDropdownEmptyStateProps) {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid={testId}
      className={cn(
        "rounded-md border border-[var(--oh-border)] px-3 py-6 text-center",
      )}
    >
      <Sparkles
        className="mx-auto mb-2 size-5 text-[var(--oh-muted)]"
        aria-hidden
      />
      <p className="text-sm font-medium text-content">
        {t(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_EMPTY_TITLE)}
      </p>
      <p className="mt-1 text-xs text-muted">
        {t(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_EMPTY_DESCRIPTION)}
      </p>
    </div>
  );
}
