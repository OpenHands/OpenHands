import { Trans, useTranslation } from "react-i18next";
import { OrgModal } from "#/components/shared/modals/org-modal";
import { I18nKey } from "#/i18n/declaration";

interface ConfirmRemoveMemberModalProps {
  onConfirm: () => void;
  onCancel: () => void;
  memberEmail: string;
  isLoading?: boolean;
}

export function ConfirmRemoveMemberModal({
  onConfirm,
  onCancel,
  memberEmail,
  isLoading = false,
}: ConfirmRemoveMemberModalProps) {
  const { t } = useTranslation();

  const confirmationMessage = (
    <Trans
      i18nKey={I18nKey.ORG$REMOVE_MEMBER_WARNING}
      values={{ email: memberEmail }}
      components={{ email: <span className="text-white" /> }}
    />
  );

  return (
    <OrgModal
      title={t(I18nKey.ORG$CONFIRM_REMOVE_MEMBER)}
      description={confirmationMessage}
      primaryButtonText={t(I18nKey.BUTTON$CONFIRM)}
      secondaryButtonText={t(I18nKey.BUTTON$CANCEL)}
      onPrimaryClick={onConfirm}
      onClose={onCancel}
      isLoading={isLoading}
      primaryButtonTestId="confirm-button"
      secondaryButtonTestId="cancel-button"
      fullWidthButtons
    />
  );
}
