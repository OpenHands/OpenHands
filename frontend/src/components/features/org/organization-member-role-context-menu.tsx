import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { ContextMenu } from "#/ui/context-menu";
import { ContextMenuListItem } from "../context-menu/context-menu-list-item";
import { ContextMenuIconText } from "#/ui/context-menu-icon-text";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { OrganizationUserRole } from "#/types/org";
import { cn } from "#/utils/utils";
import UserIcon from "#/icons/user.svg?react";
import DeleteIcon from "#/icons/u-delete.svg?react";
import AdminIcon from "#/icons/admin.svg?react";

const contextMenuListItemClassName = cn(
  "cursor-pointer p-0 h-auto hover:bg-transparent",
);

interface OrganizationMemberRoleContextMenuProps {
  onClose: () => void;
  onRoleChange: (role: OrganizationUserRole) => void;
  onRemove?: () => void;
  availableRolesToChangeTo: OrganizationUserRole[];
}

export function OrganizationMemberRoleContextMenu({
  onClose,
  onRoleChange,
  onRemove,
  availableRolesToChangeTo,
}: OrganizationMemberRoleContextMenuProps) {
  const { t } = useTranslation();
  const menuRef = useClickOutsideElement<HTMLUListElement>(onClose);

  const handleRoleChangeClick = (
    event: React.MouseEvent<HTMLButtonElement>,
    role: OrganizationUserRole,
  ) => {
    event.preventDefault();
    event.stopPropagation();
    onRoleChange(role);
    onClose();
  };

  const handleRemoveClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    onRemove?.();
    onClose();
  };

  return (
    <ContextMenu
      ref={menuRef}
      testId="organization-member-role-context-menu"
      position="bottom"
      alignment="right"
      className="min-h-fit mb-2 min-w-[195px] max-w-[195px] gap-0"
    >
      {availableRolesToChangeTo.includes("owner") && (
        <ContextMenuListItem
          testId="owner-option"
          onClick={(event) => handleRoleChangeClick(event, "owner")}
          className={contextMenuListItemClassName}
        >
          <ContextMenuIconText
            icon={
              <AdminIcon
                width={16}
                height={16}
                className="text-white pl-[2px]"
              />
            }
            text={t(I18nKey.ORG$ROLE_OWNER)}
            className="capitalize"
          />
        </ContextMenuListItem>
      )}
      {availableRolesToChangeTo.includes("admin") && (
        <ContextMenuListItem
          testId="admin-option"
          onClick={(event) => handleRoleChangeClick(event, "admin")}
          className={contextMenuListItemClassName}
        >
          <ContextMenuIconText
            icon={
              <AdminIcon
                width={16}
                height={16}
                className="text-white pl-[2px]"
              />
            }
            text={t(I18nKey.ORG$ROLE_ADMIN)}
            className="capitalize"
          />
        </ContextMenuListItem>
      )}
      {availableRolesToChangeTo.includes("member") && (
        <ContextMenuListItem
          testId="member-option"
          onClick={(event) => handleRoleChangeClick(event, "member")}
          className={contextMenuListItemClassName}
        >
          <ContextMenuIconText
            icon={<UserIcon width={16} height={16} className="text-white" />}
            text={t(I18nKey.ORG$ROLE_MEMBER)}
            className="capitalize"
          />
        </ContextMenuListItem>
      )}
      <ContextMenuListItem
        testId="remove-option"
        onClick={handleRemoveClick}
        className={contextMenuListItemClassName}
      >
        <ContextMenuIconText
          icon={<DeleteIcon width={16} height={16} className="text-red-500" />}
          text={t(I18nKey.ORG$REMOVE)}
          className="text-red-500 capitalize"
        />
      </ContextMenuListItem>
    </ContextMenu>
  );
}
