import React from "react";
import { useTranslation } from "react-i18next";
import { IoLogOutOutline, IoPersonAddOutline } from "react-icons/io5";
import { useLogout } from "#/hooks/mutation/use-logout";
import { OrganizationUserRole } from "#/types/org";
import { useOrgTypeAndAccess } from "#/hooks/use-org-type-and-access";
import { cn } from "#/utils/utils";
import { OrgSelector } from "../org/org-selector";
import { I18nKey } from "#/i18n/declaration";
import { useSettingsNavItems } from "#/hooks/use-settings-nav-items";
import DocumentIcon from "#/icons/document.svg?react";
import { ContextMenuListItem } from "../context-menu/context-menu-list-item";
import { ContextMenuContainer } from "../context-menu/context-menu-container";
import { ContextMenuCTA } from "../context-menu/context-menu-cta";
import { ContextMenuNavLink } from "../context-menu/context-menu-nav-link";
import { useShouldHideOrgSelector } from "#/hooks/use-should-hide-org-selector";
import { useBreakpoint } from "#/hooks/use-breakpoint";
import { useConfig } from "#/hooks/query/use-config";
import { ENABLE_PROJ_USER_JOURNEY } from "#/utils/feature-flags";
import { SettingsNavHeader } from "../settings/settings-nav-header";
import { SettingsNavDivider } from "../settings/settings-nav-divider";

// Shared className for context menu list items in the user context menu
const contextMenuListItemClassName = cn(
  "flex items-center gap-2 p-2 h-auto hover:bg-white/10 hover:text-white rounded text-xs",
);

interface UserContextMenuProps {
  type: OrganizationUserRole;
  onClose: () => void;
  onOpenInviteModal: () => void;
}

export function UserContextMenu({
  type,
  onClose,
  onOpenInviteModal,
}: UserContextMenuProps) {
  const { t } = useTranslation();
  const { mutate: logout } = useLogout();
  const { isPersonalOrg } = useOrgTypeAndAccess();
  const settingsNavItems = useSettingsNavItems();
  const shouldHideSelector = useShouldHideOrgSelector();
  const isMobile = useBreakpoint(768);
  const { data: config } = useConfig();

  // Keep all nav items including headers and dividers for proper section grouping
  const navItems = settingsNavItems;

  const isMember = type === "member";
  const isSaasMode = config?.app_mode === "saas";

  // Check if the ORG SETTINGS header exists in nav items
  const hasOrgHeader = navItems.some(
    (item) =>
      item.type === "header" &&
      item.text === I18nKey.SETTINGS$ORG_SETTINGS_HEADER,
  );

  // Show invite button for admin/owner in team orgs
  const showInviteButton = !isMember && !isPersonalOrg;

  // CTA only renders in SaaS desktop with feature flag enabled
  const showCta = isSaasMode && !isMobile && ENABLE_PROJ_USER_JOURNEY();
  const handleLogout = () => {
    logout();
    onClose();
  };

  const handleInviteMemberClick = () => {
    onOpenInviteModal();
    onClose();
  };

  return (
    <ContextMenuContainer testId="user-context-menu" onClose={onClose}>
      <div className="flex flex-col gap-3 w-[248px]">
        <h3 className="text-lg font-semibold text-white">
          {t(I18nKey.ORG$ACCOUNT)}
        </h3>

        <div className="flex flex-col items-start gap-0">
          {!shouldHideSelector && (
            <div className="w-full relative mb-2">
              <OrgSelector />
            </div>
          )}

          <div className="flex flex-col items-start gap-0 w-full">
            {/* Show Invite button at top if no ORG SETTINGS header exists */}
            {showInviteButton && !hasOrgHeader && (
              <ContextMenuListItem
                onClick={handleInviteMemberClick}
                className={contextMenuListItemClassName}
              >
                <IoPersonAddOutline className="text-white" size={16} />
                {t(I18nKey.ORG$INVITE_ORG_MEMBERS)}
              </ContextMenuListItem>
            )}

            {navItems.map((renderedItem, index) => {
              if (renderedItem.type === "header") {
                const isOrgHeader =
                  renderedItem.text === I18nKey.SETTINGS$ORG_SETTINGS_HEADER;
                return (
                  <React.Fragment key={`header-${renderedItem.text}`}>
                    <SettingsNavHeader
                      text={renderedItem.text}
                      className="px-2 pt-2 pb-1"
                    />
                    {/* Add Invite Organization Members right after ORG SETTINGS header */}
                    {isOrgHeader && showInviteButton && (
                      <ContextMenuListItem
                        onClick={handleInviteMemberClick}
                        className={contextMenuListItemClassName}
                      >
                        <IoPersonAddOutline className="text-white" size={16} />
                        {t(I18nKey.ORG$INVITE_ORG_MEMBERS)}
                      </ContextMenuListItem>
                    )}
                  </React.Fragment>
                );
              }

              if (renderedItem.type === "divider") {
                return (
                  <SettingsNavDivider
                    key={`divider-${index}`}
                    className="my-1.5"
                  />
                );
              }

              return (
                <ContextMenuNavLink
                  key={renderedItem.item.to}
                  item={renderedItem.item}
                  onClick={onClose}
                />
              );
            })}
          </div>

          <SettingsNavDivider className="my-1.5" />

          <a
            href="https://docs.openhands.dev"
            target="_blank"
            rel="noopener noreferrer"
            onClick={onClose}
            className="flex items-center gap-2 p-2 cursor-pointer hover:bg-white/10 hover:text-white rounded w-full text-xs"
          >
            <DocumentIcon className="text-white" width={16} height={16} />
            {t(I18nKey.SIDEBAR$DOCS)}
          </a>

          {/* Only show logout in saas mode - oss mode has no session to invalidate */}
          {isSaasMode && (
            <ContextMenuListItem
              onClick={handleLogout}
              className={contextMenuListItemClassName}
            >
              <IoLogOutOutline className="text-white" size={16} />
              {t(I18nKey.ACCOUNT_SETTINGS$LOGOUT)}
            </ContextMenuListItem>
          )}
        </div>
      </div>

      {showCta && <ContextMenuCTA />}
    </ContextMenuContainer>
  );
}
