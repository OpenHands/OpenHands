import React from "react";
import { Link } from "react-router";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { SettingsNavItem } from "#/constants/settings-nav";

interface ContextMenuNavLinkProps {
  item: SettingsNavItem;
  onClick: () => void;
}

export function ContextMenuNavLink({ item, onClick }: ContextMenuNavLinkProps) {
  const { t } = useTranslation();
  const { to, icon, text } = item;

  return (
    <Link
      to={to}
      onClick={onClick}
      className="flex items-center gap-2 p-2 cursor-pointer hover:bg-white/10 hover:text-white rounded w-full text-xs"
    >
      {React.cloneElement(icon, {
        className: "text-white",
        width: 16,
        height: 16,
        size: 16, // For react-icons compatibility
      } as React.SVGProps<SVGSVGElement>)}
      {t(text as I18nKey)}
    </Link>
  );
}
