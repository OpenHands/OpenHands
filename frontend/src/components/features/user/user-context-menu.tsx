import React from "react";
import ReactDOM from "react-dom";
import { Link, useNavigate } from "react-router";
import { useTranslation } from "react-i18next";
import {
  IoCardOutline,
  IoLogOutOutline,
  IoPersonAddOutline,
  IoPersonOutline,
} from "react-icons/io5";
import { useLogout } from "#/hooks/mutation/use-logout";
import { OrganizationUserRole } from "#/types/org";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { useOrgTypeAndAccess } from "#/hooks/use-org-type-and-access";
import { cn } from "#/utils/utils";
import { InviteOrganizationMemberModal } from "../org/invite-organization-member-modal";
import { OrgSelector } from "../org/org-selector";
import { I18nKey } from "#/i18n/declaration";
import { useSettingsNavItems } from "#/hooks/use-settings-nav-items";
import DocumentIcon from "#/icons/document.svg?react";
import { Divider } from "#/ui/divider";
import { ContextMenuListItem } from "../context-menu/context-menu-list-item";

// Shared className for context menu list items in the user context menu
// Removes default padding and hover background to match the simpler text-hover style
const contextMenuListItemClassName = cn(
  "flex items-center p-0 h-auto hover:bg-transparent hover:text-white gap-1",
);

interface UserContextMenuProps {
  type: OrganizationUserRole;
  onClose: () => void;
}

export function UserContextMenu({ type, onClose }: UserContextMenuProps) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { mutate: logout } = useLogout();
  const { isPersonalOrg } = useOrgTypeAndAccess();
  const ref = useClickOutsideElement<HTMLDivElement>(onClose);
  const settingsNavItems = useSettingsNavItems();

  // Filter out org routes since they're handled separately via buttons in this menu
  const navItems = settingsNavItems.filter(
    (item) =>
      item.to !== "/settings/org" && item.to !== "/settings/org-members",
  );

  const [inviteMemberModalIsOpen, setInviteMemberModalIsOpen] =
    React.useState(false);

  const isMember = type === "member";

  const handleLogout = () => {
    logout();
    onClose();
  };

  const handleInviteMemberClick = () => {
    setInviteMemberModalIsOpen(true);
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
        "w-64 flex flex-col gap-3 bg-tertiary border border-tertiary rounded-xl p-6",
        "text-sm absolute left-full bottom-0 z-60",
      )}
    >
      {inviteMemberModalIsOpen &&
        ReactDOM.createPortal(
          <InviteOrganizationMemberModal
            onClose={() => setInviteMemberModalIsOpen(false)}
          />,
          document.getElementById("portal-root") || document.body,
        )}

      <h3 className="text-lg font-semibold text-white">
        {t(I18nKey.ORG$ACCOUNT)}
      </h3>

      <div className="flex flex-col items-start gap-2">
        <div className="w-full relative">
          <OrgSelector />
        </div>

        {!isMember && !isPersonalOrg && (
          <>
            <ContextMenuListItem
              onClick={handleInviteMemberClick}
              className={contextMenuListItemClassName}
            >
              <IoPersonAddOutline className="text-white" size={14} />
              {t(I18nKey.ORG$INVITE_ORGANIZATION_MEMBER)}
            </ContextMenuListItem>

            <Divider className="my-1.5" />

            <ContextMenuListItem
              onClick={handleManageAccountClick}
              className={contextMenuListItemClassName}
            >
              <IoCardOutline className="text-white" size={14} />
              {t(I18nKey.ORG$MANAGE_ACCOUNT)}
            </ContextMenuListItem>
            <ContextMenuListItem
              onClick={handleManageOrganizationMembersClick}
              className={contextMenuListItemClassName}
            >
              <IoPersonOutline className="text-white" size={14} />
              {t(I18nKey.ORG$MANAGE_ORGANIZATION_MEMBERS)}
            </ContextMenuListItem>
          </>
        )}

        <Divider className="my-1.5" />

        {navItems.map((item) => (
          <Link
            key={item.to}
            to={item.to}
            onClick={onClose}
            className="flex items-center gap-1 cursor-pointer hover:text-white w-full"
          >
            {React.cloneElement(item.icon, {
              className: "text-white",
              width: 14,
              height: 14,
            } as React.SVGProps<SVGSVGElement>)}
            {t(item.text)}
          </Link>
        ))}

        <Divider className="my-1.5" />

        <a
          href="https://docs.openhands.dev"
          target="_blank"
          rel="noopener noreferrer"
          onClick={onClose}
          className="flex items-center gap-1 cursor-pointer hover:text-white w-full"
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
  );
}
