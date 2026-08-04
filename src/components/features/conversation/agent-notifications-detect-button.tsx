import { ScanSearch } from "lucide-react";
import { useTranslation } from "react-i18next";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import {
  formControlTransitionClassName,
  formControlMutedHoverClassName,
} from "#/utils/form-control-classes";

interface AgentNotificationsDetectButtonProps {
  onDetect: (event?: React.MouseEvent<HTMLButtonElement>) => void;
  disabled?: boolean;
  testId?: string;
}

export function AgentNotificationsDetectButton({
  onDetect,
  disabled = false,
  testId = "agent-notifications-detect",
}: AgentNotificationsDetectButtonProps) {
  const { t } = useTranslation("openhands");
  const label = t(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_DETECT_ARIA);

  return (
    <StyledTooltip content={label} placement="top">
      <button
        type="button"
        data-testid={testId}
        aria-label={label}
        disabled={disabled}
        onClick={(event) => onDetect(event)}
        className={cn(
          "inline-flex size-7 shrink-0 items-center justify-center rounded-lg",
          "border border-[var(--oh-border)] bg-base-secondary text-[var(--oh-muted)]",
          formControlTransitionClassName,
          formControlMutedHoverClassName,
          "disabled:cursor-not-allowed disabled:opacity-30",
        )}
      >
        <ScanSearch className="size-3.5 shrink-0" aria-hidden />
      </button>
    </StyledTooltip>
  );
}
