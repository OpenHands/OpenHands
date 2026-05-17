import React from "react";
import { Link } from "react-router";
import { useTranslation } from "react-i18next";
import { Tooltip } from "@heroui/react";
import { I18nKey } from "#/i18n/declaration";
import { SettingsNavItem } from "#/constants/settings-nav";

interface ContextMenuNavLinkProps {
  item: SettingsNavItem;
  onClick: () => void;
  disabled?: boolean;
  disabledAgentName?: string;
}

export function ContextMenuNavLink({
  item,
  onClick,
  disabled,
  disabledAgentName,
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
    const tooltip = disabledAgentName
      ? t(I18nKey.SETTINGS$AGENT_DISABLED_TOOLTIP, {
          agentName: disabledAgentName,
        })
      : undefined;
    return (
      <Tooltip content={tooltip} placement="right">
        <div className="flex items-center gap-2 p-2 opacity-40 cursor-not-allowed rounded w-full text-xs">
          {iconEl}
          {t(text as I18nKey)}
        </div>
      </Tooltip>
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
