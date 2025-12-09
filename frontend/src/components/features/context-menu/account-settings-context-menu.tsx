import React from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router";
import { ContextMenu } from "#/ui/context-menu";
import { ContextMenuListItem } from "./context-menu-list-item";
import { Divider } from "#/ui/divider";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { I18nKey } from "#/i18n/declaration";
import LogOutIcon from "#/icons/log-out.svg?react";
import DocumentIcon from "#/icons/document.svg?react";
import { useSettingsNavItems } from "#/hooks/use-settings-nav-items";

interface AccountSettingsContextMenuProps {
  onLogout: () => void;
  onClose: () => void;
}

export function AccountSettingsContextMenu({
  onLogout,
  onClose,
}: AccountSettingsContextMenuProps) {
  const ref = useClickOutsideElement<HTMLUListElement>(onClose);
  const { t } = useTranslation();
  // Get navigation items and filter out LLM settings if the feature flag is enabled
  const items = useSettingsNavItems();

  const navItems = items.map((item) => ({
    ...item,
    icon: React.cloneElement(item.icon, {
      width: 16,
      height: 16,
    } as React.SVGProps<SVGSVGElement>),
  }));
  const handleNavigationClick = () => onClose();

  return (
    <ContextMenu
      testId="account-settings-context-menu"
      ref={ref}
      alignment="right"
      className="mt-0 md:right-full md:left-full md:bottom-0 ml-0 w-fit z-[9999]"
    >
      {navItems.map(({ to, text, icon }) => (
        <Link key={to} to={to} className="text-decoration-none">
          <ContextMenuListItem
            onClick={handleNavigationClick}
            className="flex items-center gap-2 p-2 hover:bg-[#5C5D62] rounded h-[30px]"
          >
            {icon}
            <span className="text-white text-sm">{t(text)}</span>
          </ContextMenuListItem>
        </Link>
      ))}

      <Divider />

      <a
        href="https://docs.openhands.dev"
        target="_blank"
        rel="noopener noreferrer"
        className="text-decoration-none"
      >
        <ContextMenuListItem
          onClick={onClose}
          className="flex items-center gap-2 p-2 hover:bg-[#5C5D62] rounded h-[30px]"
        >
          <DocumentIcon width={16} height={16} />
          <span className="text-white text-sm">{t(I18nKey.SIDEBAR$DOCS)}</span>
        </ContextMenuListItem>
      </a>

      <ContextMenuListItem
        onClick={onLogout}
        className="flex items-center gap-2 p-2 hover:bg-[#5C5D62] rounded h-[30px]"
      >
        <LogOutIcon width={16} height={16} />
        <span className="text-white text-sm">
          {t(I18nKey.ACCOUNT_SETTINGS$LOGOUT)}
        </span>
      </ContextMenuListItem>
    </ContextMenu>
  );
}
