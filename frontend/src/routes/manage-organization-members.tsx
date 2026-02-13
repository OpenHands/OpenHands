import React from "react";
import ReactDOM from "react-dom";
import { useTranslation } from "react-i18next";
import { Plus } from "lucide-react";
import { InviteOrganizationMemberModal } from "#/components/features/org/invite-organization-member-modal";
import { ConfirmRemoveMemberModal } from "#/components/features/org/confirm-remove-member-modal";
import { ConfirmUpdateRoleModal } from "#/components/features/org/confirm-update-role-modal";
import { useOrganizationMembers } from "#/hooks/query/use-organization-members";
import { OrganizationMember, OrganizationUserRole } from "#/types/org";
import { OrganizationMemberListItem } from "#/components/features/org/organization-member-list-item";
import { useUpdateMemberRole } from "#/hooks/mutation/use-update-member-role";
import { useRemoveMember } from "#/hooks/mutation/use-remove-member";
import { useMe } from "#/hooks/query/use-me";
import { BrandButton } from "#/components/features/settings/brand-button";
import { rolePermissions } from "#/utils/org/permissions";
import { I18nKey } from "#/i18n/declaration";
import { usePermission } from "#/hooks/organizations/use-permissions";
import { getAvailableRolesAUserCanAssign } from "#/utils/org/permission-checks";
import { createPermissionGuard } from "#/utils/org/permission-guard";

export const clientLoader = createPermissionGuard(
  "invite_user_to_organization",
);

function ManageOrganizationMembers() {
  const { t } = useTranslation();
  const { data: organizationMembers } = useOrganizationMembers();
  const { data: user } = useMe();
  const { mutate: updateMemberRole, isPending: isUpdatingRole } =
    useUpdateMemberRole();
  const { mutate: removeMember, isPending: isRemovingMember } =
    useRemoveMember();
  const [inviteModalOpen, setInviteModalOpen] = React.useState(false);
  const [memberToRemove, setMemberToRemove] =
    React.useState<OrganizationMember | null>(null);
  const [memberToUpdateRole, setMemberToUpdateRole] = React.useState<{
    member: OrganizationMember;
    newRole: OrganizationUserRole;
  } | null>(null);

  const currentUserRole = user?.role ?? "member";

  const { hasPermission } = usePermission(currentUserRole);
  const hasPermissionToInvite = hasPermission("invite_user_to_organization");

  const handleRoleSelectionClick = (
    member: OrganizationMember,
    role: OrganizationUserRole,
  ) => {
    // Don't show modal if the role is the same
    if (member.role === role) {
      return;
    }
    setMemberToUpdateRole({ member, newRole: role });
  };

  const handleConfirmUpdateRole = () => {
    if (memberToUpdateRole) {
      updateMemberRole(
        {
          userId: memberToUpdateRole.member.user_id,
          role: memberToUpdateRole.newRole,
        },
        { onSettled: () => setMemberToUpdateRole(null) },
      );
    }
  };

  const handleRemoveMember = (member: OrganizationMember) => {
    setMemberToRemove(member);
  };

  const handleConfirmRemoveMember = () => {
    if (memberToRemove) {
      removeMember(
        { userId: memberToRemove.user_id },
        { onSettled: () => setMemberToRemove(null) },
      );
    }
  };

  const availableRolesToChangeTo = getAvailableRolesAUserCanAssign(
    rolePermissions[currentUserRole],
  );

  const canAssignUserRole = (member: OrganizationMember) =>
    user != null &&
    user?.user_id !== member.user_id &&
    user?.role !== member.role &&
    hasPermission(`change_user_role:${member.role}`);

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
                hasPermissionToChangeRole={canAssignUserRole(member)}
                availableRolesToChangeTo={availableRolesToChangeTo}
                onRoleChange={(role) => handleRoleSelectionClick(member, role)}
                onRemove={() => handleRemoveMember(member)}
              />
            </li>
          ))}
        </ul>
      )}

      {memberToRemove && (
        <ConfirmRemoveMemberModal
          memberEmail={memberToRemove.email}
          onConfirm={handleConfirmRemoveMember}
          onCancel={() => setMemberToRemove(null)}
          isLoading={isRemovingMember}
        />
      )}

      {memberToUpdateRole && (
        <ConfirmUpdateRoleModal
          memberEmail={memberToUpdateRole.member.email}
          newRole={memberToUpdateRole.newRole}
          onConfirm={handleConfirmUpdateRole}
          onCancel={() => setMemberToUpdateRole(null)}
          isLoading={isUpdatingRole}
        />
      )}
    </div>
  );
}

export default ManageOrganizationMembers;
