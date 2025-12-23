import React from "react";
import { useTranslation } from "react-i18next";
import { ChevronDown } from "lucide-react";
import { OrganizationMember, OrganizationUserRole } from "#/types/org";
import { cn } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";
import { OrganizationMemberRoleContextMenu } from "./organization-member-role-context-menu";

interface OrganizationMemberListItemProps {
  email: OrganizationMember["email"];
  role: OrganizationMember["role"];
  status: OrganizationMember["status"];
  hasPermissionToChangeRole: boolean;
  availableRolesToChangeTo: OrganizationUserRole[];

  onRoleChange: (role: OrganizationUserRole) => void;
  onRemove?: () => void;
}

export function OrganizationMemberListItem({
  email,
  role,
  status,
  hasPermissionToChangeRole,
  availableRolesToChangeTo,
  onRoleChange,
  onRemove,
}: OrganizationMemberListItemProps) {
  const { t } = useTranslation();
  const [contextMenuOpen, setContextMenuOpen] = React.useState(false);

  const roleSelectionIsPermitted =
    status !== "invited" && hasPermissionToChangeRole;

  const handleRoleClick = (event: React.MouseEvent<HTMLSpanElement>) => {
    if (roleSelectionIsPermitted) {
      event.preventDefault();
      event.stopPropagation();
      setContextMenuOpen(true);
    }
  };

  return (
    <div className="flex items-center justify-between py-4">
      <div className="flex items-center gap-2">
        <span
          className={cn(
            "text-sm font-semibold",
            status === "invited" && "text-gray-400",
          )}
        >
          {email}
        </span>

        {status === "invited" && (
          <span className="text-xs text-tertiary-light border border-tertiary px-2 py-1 rounded-lg">
            {t(I18nKey.ORG$STATUS_INVITED)}
          </span>
        )}
      </div>

      <div className="relative">
        <span
          onClick={handleRoleClick}
          className={cn(
            "text-xs text-gray-400 flex items-center gap-1 capitalize",
            roleSelectionIsPermitted ? "cursor-pointer" : "cursor-not-allowed",
          )}
        >
          {role}
          {hasPermissionToChangeRole && <ChevronDown size={14} />}
        </span>

        {roleSelectionIsPermitted && contextMenuOpen && (
          <OrganizationMemberRoleContextMenu
            onClose={() => setContextMenuOpen(false)}
            onRoleChange={onRoleChange}
            onRemove={onRemove}
            availableRolesToChangeTo={availableRolesToChangeTo}
          />
        )}
      </div>
    </div>
  );
}
