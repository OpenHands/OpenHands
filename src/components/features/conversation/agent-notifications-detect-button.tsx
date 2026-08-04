import { RefreshCw } from "lucide-react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { formControlButtonCompactClassName } from "#/utils/form-control-classes";

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

  return (
    <BrandButton
      testId={testId}
      type="button"
      variant="secondary"
      isDisabled={disabled}
      className={cn(formControlButtonCompactClassName, "w-7 min-w-7 px-0")}
      onClick={(event) => onDetect(event)}
      ariaLabel={t(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_DETECT_ARIA)}
    >
      <RefreshCw className="size-3.5 shrink-0" aria-hidden />
    </BrandButton>
  );
}
