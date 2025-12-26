import React from "react";
import ReactDOM from "react-dom";
import { useTranslation } from "react-i18next";
import { Plus } from "lucide-react";
import { redirect } from "react-router";
import { InviteOrganizationMemberModal } from "#/components/features/org/invite-organization-member-modal";
import { useOrganizationMembers } from "#/hooks/query/use-organization-members";
import { OrganizationUserRole } from "#/types/org";
import { OrganizationMemberListItem } from "#/components/features/org/organization-member-list-item";
import { useUpdateMemberRole } from "#/hooks/mutation/use-update-member-role";
import { useRemoveMember } from "#/hooks/mutation/use-remove-member";
import { useMe } from "#/hooks/query/use-me";
import { BrandButton } from "#/components/features/settings/brand-button";
import { rolePermissions } from "#/utils/org/permissions";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { queryClient } from "#/query-client-config";
import { getSelectedOrganizationIdFromStore } from "#/stores/selected-organization-store";
import { getMeFromQueryClient } from "#/utils/query-client-getters";
import { I18nKey } from "#/i18n/declaration";

export const clientLoader = async () => {
  const selectedOrgId = getSelectedOrganizationIdFromStore();
  let me = getMeFromQueryClient(selectedOrgId);

  if (!me && selectedOrgId) {
    me = await organizationService.getMe({ orgId: selectedOrgId });
    queryClient.setQueryData(["organizations", selectedOrgId, "me"], me);
  }

  if (!me || me.role === "user") {
    // if user is USER role, redirect to user settings
    return redirect("/settings/user");
  }

  return null;
};

function ManageOrganizationMembers() {
  const { t } = useTranslation();
  const { data: organizationMembers } = useOrganizationMembers();
  const { data: user } = useMe();
  const { mutate: updateMemberRole } = useUpdateMemberRole();
  const { mutate: removeMember } = useRemoveMember();

  const [inviteModalOpen, setInviteModalOpen] = React.useState(false);

  const currentUserRole = user?.role || "user";
  const hasPermissionToInvite = rolePermissions[currentUserRole].includes(
    "invite_user_to_organization",
  );

  const handleRoleSelectionClick = (id: string, role: OrganizationUserRole) => {
    updateMemberRole({ userId: id, role });
  };

  const handleRemoveMember = (userId: string) => {
    removeMember({ userId });
  };

  const checkIfUserHasPermissionToChangeRole = (
    memberId: string,
    memberRole: OrganizationUserRole,
  ) => {
    if (!user) return false;

    // Users cannot change their own role
    if (memberId === user.user_id) return false;

    // Owners cannot change another owner's role
    if (user.role === "owner" && memberRole === "owner") return false;

    // Admins cannot change another admin's role
    if (user.role === "admin" && memberRole === "admin") return false;

    const userPermissions = rolePermissions[user.role];
    return userPermissions.includes(`change_user_role:${memberRole}`);
  };

  const availableRolesToChangeTo = React.useMemo((): OrganizationUserRole[] => {
    if (!user) return [];
    const availableRoles: OrganizationUserRole[] = [];
    const userPermissions = rolePermissions[user.role];

    if (userPermissions.includes("change_user_role:owner")) {
      availableRoles.push("owner");
    }
    if (userPermissions.includes("change_user_role:admin")) {
      availableRoles.push("admin");
    }
    if (userPermissions.includes("change_user_role:user")) {
      availableRoles.push("user");
    }

    return availableRoles;
  }, [user]);

  return (
    <div
      data-testid="manage-organization-members-settings"
      className="px-11 py-6 flex flex-col gap-2"
    >
      {hasPermissionToInvite && (
        <BrandButton
          type="button"
          variant="secondary"
          onClick={() => setInviteModalOpen(true)}
          className="flex items-center gap-1"
        >
          <Plus size={14} />
          {t(I18nKey.ORG$INVITE_ORGANIZATION_MEMBER)}
        </BrandButton>
      )}

      {inviteModalOpen &&
        ReactDOM.createPortal(
          <InviteOrganizationMemberModal
            onClose={() => setInviteModalOpen(false)}
          />,
          document.getElementById("portal-root") || document.body,
        )}

      {organizationMembers && (
        <ul>
          {organizationMembers.map((member) => (
            <li
              key={member.user_id}
              data-testid="member-item"
              className="border-b border-tertiary"
            >
              <OrganizationMemberListItem
                email={member.email}
                role={member.role}
                status={member.status}
                hasPermissionToChangeRole={checkIfUserHasPermissionToChangeRole(
                  member.user_id,
                  member.role,
                )}
                availableRolesToChangeTo={availableRolesToChangeTo}
                onRoleChange={(role) =>
                  handleRoleSelectionClick(member.user_id, role)
                }
                onRemove={() => handleRemoveMember(member.user_id)}
              />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default ManageOrganizationMembers;
