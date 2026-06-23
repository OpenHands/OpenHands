import { useTranslation } from "react-i18next";
import { X } from "lucide-react";
import { OrganizationInvitation } from "#/types/org";
import { I18nKey } from "#/i18n/declaration";
import { CopyInviteLinkButton } from "#/components/features/org/copy-invite-link-button";

interface PendingInvitationListItemProps {
  invitation: OrganizationInvitation;
  onRevoke: () => void;
  isRevoking: boolean;
}

/**
 * A pending invitation rendered as a row in the members list, styled to
 * match OrganizationMemberListItem with an "Invited" status chip. Instead
 * of the role menu it offers copy-invite-link and revoke actions.
 */
export function PendingInvitationListItem({
  invitation,
  onRevoke,
  isRevoking,
}: PendingInvitationListItemProps) {
  const { t } = useTranslation();

  return (
    <div
      data-testid="pending-invitation-item"
      className="flex items-center justify-between py-4"
    >
      <div className="flex items-center gap-2 min-w-0">
        <span className="text-sm font-semibold leading-6 text-gray-400 truncate">
          {invitation.email}
        </span>
        <span className="text-xs text-tertiary-light border border-tertiary px-2 py-1 rounded-lg shrink-0">
          {t(I18nKey.ORG$STATUS_INVITED)}
        </span>
      </div>

      <div className="flex items-center gap-4 shrink-0">
        <span className="text-xs font-normal leading-4 text-org-text capitalize">
          {invitation.role}
        </span>
        {invitation.invite_url && (
          <CopyInviteLinkButton inviteUrl={invitation.invite_url} />
        )}
        <button
          type="button"
          data-testid="revoke-invitation-button"
          aria-label={t(I18nKey.ORG$REVOKE_INVITATION)}
          title={t(I18nKey.ORG$REVOKE_INVITATION)}
          onClick={onRevoke}
          disabled={isRevoking}
          className="text-tertiary-alt hover:text-danger cursor-pointer disabled:cursor-not-allowed"
        >
          <X size={14} />
        </button>
      </div>
    </div>
  );
}
