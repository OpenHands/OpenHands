import type { ElementType } from "react";
import { useTranslation } from "react-i18next";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import InfoCircleIcon from "#/icons/info-circle.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

interface AgentNotificationsTitleProps {
  as?: "h2" | "h3";
  className?: string;
  titleClassName?: string;
  infoTestId?: string;
}

export function AgentNotificationsTitle({
  as: Heading = "h3",
  className,
  titleClassName,
  infoTestId = "agent-notifications-info",
}: AgentNotificationsTitleProps) {
  const { t } = useTranslation("openhands");
  const HeadingTag = Heading as ElementType;

  return (
    <div className={cn("flex min-w-0 items-center gap-1.5", className)}>
      <HeadingTag
        className={cn(
          "min-w-0 text-sm font-medium text-content",
          titleClassName,
        )}
      >
        {t(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_TITLE)}
      </HeadingTag>
      <StyledTooltip
        content={
          <span className="block max-w-xs text-left">
            {t(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_DESCRIPTION)}
          </span>
        }
        placement="top"
      >
        <button
          type="button"
          data-testid={infoTestId}
          aria-label={t(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_INFO)}
          className={cn(
            "inline-flex size-5 shrink-0 items-center justify-center rounded-full",
            "text-muted transition-colors hover:bg-white/10 hover:text-white",
          )}
        >
          <InfoCircleIcon className="size-3.5" aria-hidden />
        </button>
      </StyledTooltip>
    </div>
  );
}
