import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import AutomationsIcon from "#/icons/automations.svg?react";
import { cn } from "#/utils/utils";

interface AutomationsButtonProps {
  disabled?: boolean;
}

export function AutomationsButton({
  disabled = false,
}: AutomationsButtonProps) {
  const { t } = useTranslation();

  const label = t(I18nKey.SIDEBAR$AUTOMATIONS);

  return (
    <StyledTooltip content={label} placement="right">
      <a
        href="/automations"
        data-testid="automations-button"
        aria-label={label}
        tabIndex={disabled ? -1 : 0}
        onClick={(e) => {
          if (disabled) {
            e.preventDefault();
          }
        }}
        className={cn("inline-flex items-center justify-center", {
          "pointer-events-none opacity-50": disabled,
        })}
      >
        <AutomationsIcon width={24} height={24} />
      </a>
    </StyledTooltip>
  );
}
