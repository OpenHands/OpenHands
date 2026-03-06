import React from "react";
import { useTranslation } from "react-i18next";
import { OrgModal } from "#/components/shared/modals/org-modal";
import { useInviteMembersBatch } from "#/hooks/mutation/use-invite-members-batch";
import { BadgeInput } from "#/components/shared/inputs/badge-input";
import { I18nKey } from "#/i18n/declaration";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { areAllEmailsValid, hasDuplicates } from "#/utils/input-validation";

interface InviteOrganizationMemberModalProps {
  onClose: (event?: React.MouseEvent<HTMLButtonElement>) => void;
}

export function InviteOrganizationMemberModal({
  onClose,
}: InviteOrganizationMemberModalProps) {
  const { t } = useTranslation();
  const { mutate: inviteMembers, isPending } = useInviteMembersBatch();
  const [emails, setEmails] = React.useState<string[]>([]);

  const handleEmailsChange = (newEmails: string[]) => {
    const trimmedEmails = newEmails.map((email) => email.trim());
    setEmails(trimmedEmails);
  };

  const handleSubmit = () => {
    if (emails.length === 0) {
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
      { emails },
      {
        onSuccess: () => onClose(),
      },
    );
  };

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
      <BadgeInput
        name="emails-badge-input"
        value={emails}
        placeholder={t(I18nKey.COMMON$TYPE_EMAIL_AND_PRESS_SPACE)}
        onChange={handleEmailsChange}
      />
    </OrgModal>
  );
}
