import React from "react";
import { Link } from "react-router";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { SettingsNavItem } from "#/constants/settings-nav";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";

interface ContextMenuNavLinkProps {
  item: SettingsNavItem;
  onClick: () => void;
  disabled?: boolean;
  disabledReason?: string;
}

export function ContextMenuNavLink({
  item,
  onClick,
  disabled,
  disabledReason,
}: ContextMenuNavLinkProps) {
  const { t } = useTranslation();
  const { to, icon, text } = item;

  const iconEl = React.cloneElement(icon, {
    className: "text-white",
    width: 16,
    height: 16,
    size: 16,
  } as React.SVGProps<SVGSVGElement>);

  if (disabled) {
    const tooltip = disabledReason
      ? t(I18nKey.SETTINGS$AGENT_DISABLED_TOOLTIP, {
          agentName: disabledReason,
        })
      : undefined;
    const inner = (
      <div className="flex items-center gap-2 p-2 rounded w-full text-xs opacity-40 cursor-not-allowed">
        {iconEl}
        {t(text as I18nKey)}
      </div>
    );
    return tooltip ? (
      <StyledTooltip content={tooltip}>{inner}</StyledTooltip>
    ) : (
      inner
    );
  }

  return (
    <Link
      to={to}
      onClick={onClick}
      className="flex items-center gap-2 p-2 cursor-pointer hover:bg-white/10 hover:text-white rounded w-full text-xs"
    >
      {iconEl}
      {t(text as I18nKey)}
    </Link>
  );
}
