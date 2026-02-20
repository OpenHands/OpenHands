import { Trans, useTranslation } from "react-i18next";
import { OrgModal } from "#/components/shared/modals/org-modal";
import { I18nKey } from "#/i18n/declaration";
import { OrganizationUserRole } from "#/types/org";

interface ConfirmUpdateRoleModalProps {
  onConfirm: () => void;
  onCancel: () => void;
  memberEmail: string;
  newRole: OrganizationUserRole;
  isLoading?: boolean;
}

export function ConfirmUpdateRoleModal({
  onConfirm,
  onCancel,
  memberEmail,
  newRole,
  isLoading = false,
}: ConfirmUpdateRoleModalProps) {
  const { t } = useTranslation();

  const confirmationMessage = (
    <Trans
      i18nKey={I18nKey.ORG$UPDATE_ROLE_WARNING}
      values={{ email: memberEmail, role: newRole }}
      components={{
        email: <span className="text-white" />,
        role: <span className="text-white capitalize" />,
      }}
    />
  );

  return (
    <OrgModal
      title={t(I18nKey.ORG$CONFIRM_UPDATE_ROLE)}
      description={confirmationMessage}
      primaryButtonText={t(I18nKey.BUTTON$CONFIRM)}
      onPrimaryClick={onConfirm}
      onClose={onCancel}
      isLoading={isLoading}
      primaryButtonTestId="confirm-button"
      secondaryButtonTestId="cancel-button"
      fullWidthButtons
    />
  );
}
