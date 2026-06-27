import React from "react";
import { useTranslation } from "react-i18next";
import { OrgModal } from "#/components/shared/modals/org-modal";
import { useInviteMembersBatch } from "#/hooks/mutation/use-invite-members-batch";
import { BadgeInput } from "#/components/shared/inputs/badge-input";
import { I18nKey } from "#/i18n/declaration";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { areAllEmailsValid, hasDuplicates } from "#/utils/input-validation";
import { Dropdown } from "#/ui/dropdown/dropdown";
import { BatchInvitationResult, OrganizationUserRole } from "#/types/org";
import { CopyInviteLinkButton } from "#/components/features/org/copy-invite-link-button";
import { usePendingInvitations } from "#/hooks/query/use-pending-invitations";

interface InviteOrganizationMemberModalProps {
  onClose: (event?: React.MouseEvent<HTMLButtonElement>) => void;
}

export function InviteOrganizationMemberModal({
  onClose,
}: InviteOrganizationMemberModalProps) {
  const { t } = useTranslation();
  const { mutate: inviteMembers, isPending } = useInviteMembersBatch();
  // The modal is only reachable with the invite permission, which is what
  // gates the backing endpoint; the response carries auto_add_enabled.
  const { data: pendingData } = usePendingInvitations(true);
  const [emails, setEmails] = React.useState<string[]>([]);
  const [role, setRole] = React.useState<OrganizationUserRole>("member");
  const [result, setResult] = React.useState<BatchInvitationResult | null>(
    null,
  );

  const handleEmailsChange = (newEmails: string[]) => {
    const trimmedEmails = newEmails.map((email) => email.trim());
    setEmails(trimmedEmails);
  };

  const handleSubmit = () => {
    if (emails.length === 0) {
      displayErrorToast(t(I18nKey.ORG$NO_EMAILS_ADDED_HINT));
      return;
    }

    if (!areAllEmailsValid(emails)) {
      displayErrorToast(t(I18nKey.SETTINGS$INVALID_EMAIL_FORMAT));
      return;
    }

    if (hasDuplicates(emails)) {
      displayErrorToast(t(I18nKey.ORG$DUPLICATE_EMAILS_ERROR));
      return;
    }

    inviteMembers(
      { emails, role },
      {
        onSuccess: (data) => {
          // When email delivery works, the invitees are notified and the
          // modal can simply close. Without it, the links are the only way
          // invitees can ever join — keep the modal open so the inviter can
          // copy them.
          if (data.email_delivery_configured && data.failed.length === 0) {
            onClose();
          } else {
            setResult(data);
          }
        },
      },
    );
  };

  const roleOptions = [
    { value: "member", label: t(I18nKey.ORG$ROLE_MEMBER) },
    { value: "admin", label: t(I18nKey.ORG$ROLE_ADMIN) },
  ];

  if (result) {
    return (
      <OrgModal
        testId="invite-links-modal"
        title={t(I18nKey.ORG$INVITE_ORG_MEMBERS)}
        description={
          result.email_delivery_configured
            ? undefined
            : t(I18nKey.ORG$EMAIL_DELIVERY_NOT_CONFIGURED)
        }
        primaryButtonText={t(I18nKey.BUTTON$CLOSE)}
        onPrimaryClick={() => onClose()}
        onClose={onClose}
        hideSecondaryButton
      >
        <div className="flex flex-col gap-2" data-testid="invite-links-list">
          {/* With email configured the modal only stays open to show failures;
              "share these links" is only accurate when no email was sent. */}
          {!result.email_delivery_configured &&
            result.successful.length > 0 && (
              <span className="text-sm">
                {t(I18nKey.ORG$INVITATIONS_CREATED_SHARE_LINKS)}
              </span>
            )}
          {result.successful.map((invitation) => (
            <div
              key={invitation.id}
              className="flex items-center justify-between gap-2 text-sm"
            >
              <span className="truncate">{invitation.email}</span>
              {invitation.invite_url && (
                <CopyInviteLinkButton inviteUrl={invitation.invite_url} />
              )}
            </div>
          ))}
          {result.failed.map((failure) => (
            <div
              key={failure.email}
              className="flex items-center justify-between gap-2 text-sm text-danger"
            >
              <span className="truncate">{failure.email}</span>
              <span className="truncate">{failure.error}</span>
            </div>
          ))}
        </div>
      </OrgModal>
    );
  }

  return (
    <OrgModal
      testId="invite-modal"
      title={t(I18nKey.ORG$INVITE_ORG_MEMBERS)}
      description={t(I18nKey.ORG$INVITE_USERS_DESCRIPTION)}
      primaryButtonText={t(I18nKey.BUTTON$ADD)}
      onPrimaryClick={handleSubmit}
      onClose={onClose}
      isLoading={isPending}
    >
      {pendingData?.auto_add_enabled && (
        <p
          data-testid="auto-add-enabled-hint"
          className="text-xs text-tertiary-alt"
        >
          {t(I18nKey.ORG$AUTO_ADD_ENABLED_HINT)}
        </p>
      )}
      <BadgeInput
        name="emails-badge-input"
        value={emails}
        placeholder={t(I18nKey.COMMON$ENTER_EMAIL_ADDRESSES)}
        onChange={handleEmailsChange}
      />
      <label className="flex flex-col gap-1 text-sm capitalize">
        {t(I18nKey.ORG$INVITE_ROLE_LABEL)}
        <Dropdown
          testId="invite-role-dropdown"
          options={roleOptions}
          defaultValue={roleOptions[0]}
          onChange={(option) =>
            setRole((option?.value as OrganizationUserRole) ?? "member")
          }
        />
      </label>
    </OrgModal>
  );
}
