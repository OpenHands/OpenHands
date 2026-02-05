import { NavLink } from "react-router";
import { useTranslation } from "react-i18next";
import OpenHandsLogo from "#/assets/branding/openhands-logo.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";

export function OpenHandsLogoButton() {
  const { t } = useTranslation();

  const tooltipText = t(I18nKey.BRANDING$OPENHANDS);
  const ariaLabel = t(I18nKey.BRANDING$OPENHANDS_LOGO);

  return (
    <StyledTooltip content={tooltipText}>
      <NavLink to="/" aria-label={ariaLabel}>
        <OpenHandsLogo width={46} height={30} />
      </NavLink>
    </StyledTooltip>
  );
}
