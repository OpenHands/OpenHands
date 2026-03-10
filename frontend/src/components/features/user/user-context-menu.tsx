import React from "react";
import { Link, useNavigate } from "react-router";
import { useTranslation } from "react-i18next";
import {
  IoCardOutline,
  IoLogOutOutline,
  IoPersonAddOutline,
} from "react-icons/io5";
import { FiUsers } from "react-icons/fi";
import { useLogout } from "#/hooks/mutation/use-logout";
import { OrganizationUserRole } from "#/types/org";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { useOrgTypeAndAccess } from "#/hooks/use-org-type-and-access";
import { cn } from "#/utils/utils";
import { OrgSelector } from "../org/org-selector";
import { I18nKey } from "#/i18n/declaration";
import { useSettingsNavItems } from "#/hooks/use-settings-nav-items";
import DocumentIcon from "#/icons/document.svg?react";
import { Divider } from "#/ui/divider";
import { ContextMenuListItem } from "../context-menu/context-menu-list-item";
import { useShouldHideOrgSelector } from "#/hooks/use-should-hide-org-selector";

// Shared className for context menu list items in the user context menu
const contextMenuListItemClassName = cn(
  "flex items-center gap-2 p-2 h-auto hover:bg-white/10 hover:text-white rounded",
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
  const navigate = useNavigate();
  const { mutate: logout } = useLogout();
  const { isPersonalOrg } = useOrgTypeAndAccess();
  const ref = useClickOutsideElement<HTMLDivElement>(onClose);
  const settingsNavItems = useSettingsNavItems();
  const shouldHideSelector = useShouldHideOrgSelector();

  // Filter out org routes since they're handled separately via buttons in this menu
  const navItems = settingsNavItems.filter(
    (item) =>
      item.to !== "/settings/org" && item.to !== "/settings/org-members",
  );

  const isMember = type === "member";

  const handleLogout = () => {
    logout();
    onClose();
  };

  const handleInviteMemberClick = () => {
    onOpenInviteModal();
    onClose();
  };

  const handleManageOrganizationMembersClick = () => {
    navigate("/settings/org-members");
    onClose();
  };

  const handleManageAccountClick = () => {
    navigate("/settings/org");
    onClose();
  };

  return (
    <div
      data-testid="user-context-menu"
      ref={ref}
      className={cn(
        "w-72 flex flex-col gap-3 bg-tertiary border border-tertiary rounded-xl p-4 context-menu-box-shadow",
        "text-sm absolute left-full bottom-0 z-101",
      )}
    >
      <h3 className="text-lg font-semibold text-white">
        {t(I18nKey.ORG$ACCOUNT)}
      </h3>

      <div className="flex flex-col items-start gap-2">
        {!shouldHideSelector && (
          <div className="w-full relative">
            <OrgSelector />
          </div>
        )}

        {!isMember && !isPersonalOrg && (
          <div className="flex flex-col items-start gap-0 w-full">
            <ContextMenuListItem
              onClick={handleInviteMemberClick}
              className={contextMenuListItemClassName}
            >
              <IoPersonAddOutline className="text-white" size={14} />
              {t(I18nKey.ORG$INVITE_ORG_MEMBERS)}
            </ContextMenuListItem>

            <Divider className="my-1.5" />

            <ContextMenuListItem
              onClick={handleManageAccountClick}
              className={contextMenuListItemClassName}
            >
              <IoCardOutline className="text-white" size={14} />
              {t(I18nKey.COMMON$ORGANIZATIONS)}
            </ContextMenuListItem>
            <ContextMenuListItem
              onClick={handleManageOrganizationMembersClick}
              className={contextMenuListItemClassName}
            >
              <FiUsers className="text-white shrink-0" size={14} />
              {t(I18nKey.ORG$ORGANIZATION_MEMBERS)}
            </ContextMenuListItem>
            <Divider className="my-1.5" />
          </div>
        )}

        <div className="flex flex-col items-start gap-0 w-full">
          {navItems.map((item) => (
            <Link
              key={item.to}
              to={item.to}
              onClick={onClose}
              className="flex items-center gap-2 p-2 cursor-pointer hover:bg-white/10 hover:text-white rounded w-full"
            >
              {React.cloneElement(item.icon, {
                className: "text-white",
                width: 14,
                height: 14,
              } as React.SVGProps<SVGSVGElement>)}
              {t(item.text)}
            </Link>
          ))}
        </div>

        <Divider className="my-1.5" />

        <div className="flex flex-col items-start gap-0 w-full">
          <a
            href="https://docs.openhands.dev"
            target="_blank"
            rel="noopener noreferrer"
            onClick={onClose}
            className="flex items-center gap-2 p-2 cursor-pointer hover:bg-white/10 hover:text-white rounded w-full"
          >
            <DocumentIcon className="text-white" width={14} height={14} />
            {t(I18nKey.SIDEBAR$DOCS)}
          </a>

          <ContextMenuListItem
            onClick={handleLogout}
            className={contextMenuListItemClassName}
          >
            <IoLogOutOutline className="text-white" size={14} />
            {t(I18nKey.ACCOUNT_SETTINGS$LOGOUT)}
          </ContextMenuListItem>
        </div>
      </div>
    </div>
  );
}
