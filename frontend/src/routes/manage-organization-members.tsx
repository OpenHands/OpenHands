import React from "react";
import ReactDOM from "react-dom";
import { useTranslation } from "react-i18next";
import { LoaderCircle, Plus, Search } from "lucide-react";
import { InviteOrganizationMemberModal } from "#/components/features/org/invite-organization-member-modal";
import { ConfirmRemoveMemberModal } from "#/components/features/org/confirm-remove-member-modal";
import { ConfirmUpdateRoleModal } from "#/components/features/org/confirm-update-role-modal";
import { useOrganizationMembers } from "#/hooks/query/use-organization-members";
import { useOrganizationMembersCount } from "#/hooks/query/use-organization-members-count";
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
import { Typography } from "#/ui/typography";
import { Pagination } from "#/ui/pagination";
import { useDebounce } from "#/hooks/use-debounce";

export const clientLoader = createPermissionGuard(
  "invite_user_to_organization",
);

export const handle = { hideTitle: true };

function ManageOrganizationMembers() {
  const { t } = useTranslation();

  // Pagination and filtering state
  const [page, setPage] = React.useState(1);
  const [emailFilter, setEmailFilter] = React.useState("");
  const debouncedEmailFilter = useDebounce(emailFilter, 300);

  // Reset to page 1 when filter changes
  React.useEffect(() => {
    setPage(1);
  }, [debouncedEmailFilter]);

  const limit = 10;

  const {
    data: membersData,
    isLoading,
    isFetching,
    error: membersError,
  } = useOrganizationMembers({
    page,
    limit,
    email: debouncedEmailFilter,
  });

  const { data: totalCount, error: countError } = useOrganizationMembersCount({
    email: debouncedEmailFilter,
  });

  const hasError = membersError || countError;

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

  // Calculate total pages
  const totalPages =
    totalCount !== undefined ? Math.ceil(totalCount / limit) : 0;

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
      className="flex flex-col gap-2 h-full"
    >
      <div className="flex items-center justify-between pb-6">
        <Typography.H2>{t(I18nKey.ORG$ORGANIZATION_MEMBERS)}</Typography.H2>
        {hasPermissionToInvite && (
          <BrandButton
            type="button"
            variant="secondary"
            onClick={() => setInviteModalOpen(true)}
            className="flex items-center gap-1 rounded-full border-none text-sm text-white font-medium leading-5 bg-org-button hover:opacity-80 p-2"
          >
            <Plus size={14} />
            {t(I18nKey.ORG$INVITE_ORG_MEMBERS)}
          </BrandButton>
        )}
      </div>

      {/* Email Search Input */}
      <div className="rounded w-80 p-3 placeholder:text-tertiary-alt bg-modal-input border-none flex items-center gap-2 mb-4">
        <Search size={16} className="text-tertiary-alt" />
        <input
          data-testid="email-filter-input"
          type="text"
          value={emailFilter}
          placeholder={t(I18nKey.ORG$SEARCH_BY_EMAIL)}
          onChange={(e) => setEmailFilter(e.target.value)}
          className="w-full text-sm leading-4 font-normal outline-none bg-transparent"
        />
        {isFetching && debouncedEmailFilter && (
          <LoaderCircle
            size={16}
            className="text-tertiary-alt animate-spin"
            data-testid="search-loading-indicator"
          />
        )}
      </div>

      {inviteModalOpen &&
        ReactDOM.createPortal(
          <InviteOrganizationMemberModal
            onClose={() => setInviteModalOpen(false)}
          />,
          document.getElementById("portal-root") || document.body,
        )}

      <div className="rounded-xl border border-org-border bg-org-background table-box-shadow flex-1 overflow-y-auto custom-scrollbar">
        <div className="flex items-center justify-between pl-6 pr-6 text-[11px] text-white font-medium leading-4 border-b border-org-divider w-full h-9">
          <span>{t(I18nKey.ORG$ALL_ORGANIZATION_MEMBERS)}</span>
          {totalCount !== undefined && (
            <span className="text-tertiary-alt">
              {totalCount} {totalCount === 1 ? "member" : "members"}
            </span>
          )}
        </div>

        {isLoading && (
          <div className="flex items-center justify-center p-8 text-tertiary-alt">
            Loading...
          </div>
        )}

        {!isLoading && hasError && (
          <div className="flex items-center justify-center p-8 text-tertiary-alt">
            {t(I18nKey.ORG$FAILED_TO_LOAD_MEMBERS)}
          </div>
        )}

        {!isLoading &&
          !hasError &&
          membersData?.items &&
          membersData.items.length > 0 && (
            <ul>
              {membersData.items.map((member) => (
                <li
                  key={member.user_id}
                  data-testid="member-item"
                  className="border-b border-org-divider last:border-none px-6"
                >
                  <OrganizationMemberListItem
                    email={member.email}
                    role={member.role}
                    status={member.status}
                    hasPermissionToChangeRole={canAssignUserRole(member)}
                    availableRolesToChangeTo={availableRolesToChangeTo}
                    onRoleChange={(role) =>
                      handleRoleSelectionClick(member, role)
                    }
                    onRemove={() => handleRemoveMember(member)}
                  />
                </li>
              ))}
            </ul>
          )}

        {!isLoading &&
          !hasError &&
          (!membersData?.items || membersData.items.length === 0) && (
            <div className="flex items-center justify-center p-8 text-tertiary-alt">
              {debouncedEmailFilter
                ? t(I18nKey.ORG$NO_MEMBERS_MATCHING_FILTER)
                : t(I18nKey.ORG$NO_MEMBERS_FOUND)}
            </div>
          )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <Pagination
          currentPage={page}
          totalPages={totalPages}
          onPageChange={setPage}
          className="py-4"
        />
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
