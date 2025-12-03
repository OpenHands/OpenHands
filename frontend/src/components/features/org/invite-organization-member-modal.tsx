import React from "react";
import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { useInviteMembersBatch } from "#/hooks/mutation/use-invite-members-batch";
import { BrandButton } from "../settings/brand-button";
import { BadgeInput } from "#/components/shared/inputs/badge-input";
import { I18nKey } from "#/i18n/declaration";

interface InviteOrganizationMemberModalProps {
  onClose: (event?: React.MouseEvent<HTMLButtonElement>) => void;
}

export function InviteOrganizationMemberModal({
  onClose,
}: InviteOrganizationMemberModalProps) {
  const { t } = useTranslation();
  const { mutate: inviteMembers } = useInviteMembersBatch();
  const [emails, setEmails] = React.useState<string[]>([]);

  const formAction = () => {
    if (emails.length > 0) {
      inviteMembers({ emails });
      onClose();
    }
  };

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        data-testid="invite-modal"
        className="bg-base rounded-xl p-4 border w-sm border-tertiary items-start"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="w-full flex flex-col gap-2">
          <h3 className="text-lg font-semibold">
            {t(I18nKey.ORG$INVITE_USERS)}
          </h3>
          <p className="text-xs text-gray-400">
            {t(I18nKey.ORG$INVITE_USERS_DESCRIPTION)}
          </p>
          <div className="flex flex-col gap-2">
            <span className="text-sm">{t(I18nKey.ORG$EMAILS)}</span>
            <BadgeInput
              name="emails-badge-input"
              value={emails}
              placeholder="Type email and press space"
              onChange={setEmails}
            />
          </div>

          <div className="flex gap-2">
            <BrandButton
              type="button"
              variant="primary"
              className="flex-1"
              onClick={formAction}
            >
              {t(I18nKey.BUTTON$ADD)}
            </BrandButton>
            <BrandButton
              type="button"
              variant="secondary"
              onClick={onClose}
              className="flex-1"
            >
              {t(I18nKey.BUTTON$CANCEL)}
            </BrandButton>
          </div>
        </div>
      </div>
    </ModalBackdrop>
  );
}
