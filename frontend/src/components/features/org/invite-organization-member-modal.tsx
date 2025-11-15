import React from "react";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { useInviteMembersBatch } from "#/hooks/mutation/use-invite-members-batch";
import { BrandButton } from "../settings/brand-button";
import { BadgeInput } from "#/components/shared/inputs/badge-input";

interface InviteOrganizationMemberModalProps {
  onClose: (event?: React.MouseEvent<HTMLButtonElement>) => void;
}

export function InviteOrganizationMemberModal({
  onClose,
}: InviteOrganizationMemberModalProps) {
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
          <h3 className="text-lg font-semibold">Invite Users</h3>
          <p className="text-xs text-gray-400">
            Invite colleaguess using their email address
          </p>
          <div className="flex flex-col gap-2">
            <span className="text-sm">Emails</span>
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
              Next
            </BrandButton>
            <BrandButton
              type="button"
              variant="secondary"
              onClick={onClose}
              className="flex-1"
            >
              Skip
            </BrandButton>
          </div>
        </div>
      </div>
    </ModalBackdrop>
  );
}
