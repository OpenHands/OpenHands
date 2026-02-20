import { NavLink } from "react-router";
import { useTranslation } from "react-i18next";
import b1Logo from "#/assets/branding/b1-logo.svg";
import { I18nKey } from "#/i18n/declaration";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";

export function OpenHandsLogoButton() {
  const { t } = useTranslation();

  const tooltipText = t(I18nKey.BRANDING$OPENHANDS);
  const ariaLabel = t(I18nKey.BRANDING$OPENHANDS_LOGO);

  return (
    <StyledTooltip content={tooltipText}>
      <NavLink to="/" aria-label={ariaLabel}>
        <img src={b1Logo} alt="Build.One" width={46} height={30} />
      </NavLink>
    </StyledTooltip>
  );
}
