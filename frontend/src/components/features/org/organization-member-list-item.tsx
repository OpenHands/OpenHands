import React from "react";
import { ChevronDown } from "lucide-react";
import { OrganizationMember, OrganizationUserRole } from "#/types/org";
import { cn } from "#/utils/utils";

interface OrganizationMemberListItemProps {
  email: OrganizationMember["email"];
  role: OrganizationMember["role"];
  status: OrganizationMember["status"];
  hasPermissionToChangeRole: boolean;

  onRoleChange: (role: OrganizationUserRole) => void;
  onRemove?: () => void;
}

export function OrganizationMemberListItem({
  email,
  role,
  status,
  hasPermissionToChangeRole,
  onRoleChange,
  onRemove,
}: OrganizationMemberListItemProps) {
  const [roleSelectionOpen, setRoleSelectionOpen] = React.useState(false);

  const handleRoleSelectionClick = (newRole: OrganizationUserRole) => {
    onRoleChange(newRole);
    setRoleSelectionOpen(false);
  };

  const roleSelectionIsPermitted =
    status !== "invited" && hasPermissionToChangeRole;

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
            invited
          </span>
        )}
      </div>
      <span
        onClick={() => setRoleSelectionOpen(true)}
        className={cn(
          "text-xs text-gray-400 uppercase flex items-center gap-1",
          roleSelectionIsPermitted ? "cursor-pointer" : "cursor-not-allowed",
        )}
      >
        {role}
        {hasPermissionToChangeRole && <ChevronDown size={14} />}
      </span>

      {roleSelectionIsPermitted && roleSelectionOpen && (
        <ul data-testid="role-dropdown">
          <li>
            <span onClick={() => handleRoleSelectionClick("admin")}>admin</span>
          </li>
          <li>
            <span onClick={() => handleRoleSelectionClick("user")}>user</span>
          </li>
          <li>
            <span 
              className="text-red-500 cursor-pointer" 
              onClick={() => {
                onRemove?.();
                setRoleSelectionOpen(false);
              }}
            >
              remove
            </span>
          </li>
        </ul>
      )}
    </div>
  );
}
