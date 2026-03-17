import { Trans, useTranslation } from "react-i18next";
import { OrgModal } from "#/components/shared/modals/org-modal";
import { I18nKey } from "#/i18n/declaration";
import { useDeleteOrganization } from "#/hooks/mutation/use-delete-organization";
import { useOrganization } from "#/hooks/query/use-organization";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

interface DeleteOrgConfirmationModalProps {
  onClose: () => void;
}

export function DeleteOrgConfirmationModal({
  onClose,
}: DeleteOrgConfirmationModalProps) {
  const { t } = useTranslation();
  const { mutate: deleteOrganization, isPending } = useDeleteOrganization();
  const { data: organization } = useOrganization();

  const handleConfirm = () => {
    deleteOrganization(undefined, {
      onSuccess: onClose,
      onError: () => {
        displayErrorToast(t(I18nKey.ORG$DELETE_ORGANIZATION_ERROR));
      },
    });
  };

  const confirmationMessage = organization?.name ? (
    <Trans
      i18nKey={I18nKey.ORG$DELETE_ORGANIZATION_WARNING_WITH_NAME}
      values={{ name: organization.name }}
      components={{ name: <span className="text-white" /> }}
    />
  ) : (
    t(I18nKey.ORG$DELETE_ORGANIZATION_WARNING)
  );

  return (
    <OrgModal
      testId="delete-org-confirmation"
      title={t(I18nKey.ORG$DELETE_ORGANIZATION)}
      description={confirmationMessage}
      primaryButtonText={t(I18nKey.BUTTON$CONFIRM)}
      onPrimaryClick={handleConfirm}
      onClose={onClose}
      isLoading={isPending}
      secondaryButtonTestId="cancel-button"
      ariaLabel={t(I18nKey.ORG$DELETE_ORGANIZATION)}
      fullWidthButtons
    />
  );
}
