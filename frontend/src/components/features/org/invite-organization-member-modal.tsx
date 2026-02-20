import React from "react";
import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { useInviteMembersBatch } from "#/hooks/mutation/use-invite-members-batch";
import { BrandButton } from "../settings/brand-button";
import { BadgeInput } from "#/components/shared/inputs/badge-input";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
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
    // Trim emails to avoid whitespace issues from copy-paste
    const trimmedEmails = newEmails.map((email) => email.trim());
    setEmails(trimmedEmails);
  };

  const formAction = () => {
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
    <ModalBackdrop onClose={isPending ? undefined : onClose}>
      <div
        data-testid="invite-modal"
        className="bg-[#171717] rounded-xl p-6 w-sm items-start modal-box-shadow"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="w-full flex flex-col gap-2">
          <h3 className="text-xl leading-6 font-semibold">
            {t(I18nKey.ORG$INVITE_USERS)}
          </h3>
          <p className="text-xs leading-4 font-normal text-[#A3A3A3]">
            {t(I18nKey.ORG$INVITE_USERS_DESCRIPTION)}
          </p>
          <div className="flex flex-col gap-2">
            <BadgeInput
              name="emails-badge-input"
              value={emails}
              placeholder={t(I18nKey.COMMON$TYPE_EMAIL_AND_PRESS_SPACE)}
              onChange={handleEmailsChange}
              className="bg-[#27272A] border-none pl-3"
              inputClassName="text-sm leading-4 font-normal"
            />
          </div>

          <div className="flex gap-2 pt-4">
            <BrandButton
              type="button"
              variant="primary"
              className="flex-1 flex items-center justify-center bg-[#F3CE49] text-sm leading-4 font-medium rounded h-10"
              onClick={formAction}
              isDisabled={isPending}
            >
              {isPending ? (
                <LoadingSpinner
                  size="small"
                  className="w-5 h-5"
                  innerClassName="hidden"
                  outerClassName="w-5 h-5"
                />
              ) : (
                t(I18nKey.BUTTON$ADD)
              )}
            </BrandButton>
            <BrandButton
              type="button"
              variant="secondary"
              onClick={onClose}
              className="flex-1 bg-[#737373] text-sm text-white leading-4 font-medium rounded border-none h-10"
              isDisabled={isPending}
            >
              {t(I18nKey.BUTTON$CLOSE)}
            </BrandButton>
          </div>
        </div>
      </div>
    </ModalBackdrop>
  );
}
