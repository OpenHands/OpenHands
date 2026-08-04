import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import type { AgentNotificationKind } from "./agent-notifications.constants";

const AGENT_NOTIFICATION_KIND_PILL_CLASS_NAME = cn(
  "inline-flex w-fit shrink-0 items-center whitespace-nowrap rounded-full",
  "border border-[var(--oh-border)] bg-[var(--oh-surface)] px-2 py-0.5",
  "text-[11px] font-medium leading-4 text-tertiary-light",
);

const AUTOMATION_TYPE_LABEL_KEY: Record<
  Exclude<AgentNotificationKind, "skill">,
  I18nKey
> = {
  workflow: I18nKey.AUTOMATE$DASHBOARD_KIND_WORKFLOW,
  routine: I18nKey.AUTOMATE$DASHBOARD_KIND_ROUTINE,
  responder: I18nKey.AUTOMATE$DASHBOARD_KIND_RESPONDER,
};

interface AgentNotificationKindPillProps {
  kind: AgentNotificationKind;
  testId?: string;
}

export function AgentNotificationKindPill({
  kind,
  testId,
}: AgentNotificationKindPillProps) {
  const { t } = useTranslation("openhands");

  const label =
    kind === "skill"
      ? t(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_KIND_SKILL)
      : t(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_AUTOMATION_TYPE_PILL, {
          type: t(AUTOMATION_TYPE_LABEL_KEY[kind]),
        });

  return (
    <span
      data-testid={testId}
      className={AGENT_NOTIFICATION_KIND_PILL_CLASS_NAME}
    >
      {label}
    </span>
  );
}
