import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { usePendingInvitations } from "#/hooks/query/use-pending-invitations";
import { CopyInviteLinkButton } from "#/components/features/org/copy-invite-link-button";

/**
 * Pending invitations for the current org with copyable invite links.
 * Rendered only for users with the invite permission — the backing
 * endpoint is gated on it and the links carry the invitation tokens.
 */
export function PendingInvitationsList() {
  const { t } = useTranslation();
  const { data, isLoading } = usePendingInvitations(true);

  if (isLoading || !data || data.items.length === 0) {
    return null;
  }

  return (
    <div
      data-testid="pending-invitations-list"
      className="rounded-xl border border-org-border bg-org-background table-box-shadow overflow-y-auto custom-scrollbar mt-4 shrink-0 max-h-60"
    >
      <div className="flex items-center justify-between pl-6 pr-6 text-[11px] text-white font-medium leading-4 border-b border-org-divider w-full h-9">
        <span>{t(I18nKey.ORG$PENDING_INVITATIONS)}</span>
        <span className="text-tertiary-alt">{data.items.length}</span>
      </div>

      {!data.email_delivery_configured && (
        <div
          data-testid="email-not-configured-hint"
          className="px-6 py-2 text-xs text-tertiary-alt border-b border-org-divider"
        >
          {t(I18nKey.ORG$EMAIL_DELIVERY_NOT_CONFIGURED)}
        </div>
      )}

      {data.items.map((invitation) => (
        <div
          key={invitation.id}
          className="flex items-center justify-between pl-6 pr-6 py-2 text-sm border-b border-org-divider last:border-b-0"
        >
          <div className="flex items-center gap-3 min-w-0">
            <span className="truncate">{invitation.email}</span>
            <span className="text-xs text-tertiary-alt uppercase">
              {invitation.role}
            </span>
          </div>
          {invitation.invite_url && (
            <CopyInviteLinkButton inviteUrl={invitation.invite_url} />
          )}
        </div>
      ))}
    </div>
  );
}
