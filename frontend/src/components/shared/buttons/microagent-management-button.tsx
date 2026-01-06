import { NavLink } from "react-router";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import RobotIcon from "#/icons/robot.svg?react";

interface MicroagentManagementButtonProps {
  disabled?: boolean;
}

export function MicroagentManagementButton({
  disabled = false,
}: MicroagentManagementButtonProps) {
  const { t } = useTranslation();

  const microagentManagement = t(I18nKey.MICROAGENT_MANAGEMENT$TITLE);

  return (
    <StyledTooltip content={microagentManagement}>
      <NavLink
        to="/microagent-management"
        data-testid="microagent-management-button"
        aria-label={microagentManagement}
        tabIndex={disabled ? -1 : 0}
        onClick={(e) => {
          if (disabled) {
            e.preventDefault();
          }
        }}
        className={disabled ? "pointer-events-none opacity-50" : undefined}
      >
        <RobotIcon width={28} height={28} />
      </NavLink>
    </StyledTooltip>
  );
}
