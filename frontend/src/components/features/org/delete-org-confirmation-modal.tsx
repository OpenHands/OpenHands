import { Trans, useTranslation } from "react-i18next";
import {
  BaseModalDescription,
  BaseModalTitle,
} from "#/components/shared/modals/confirmation-modals/base-modal";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { I18nKey } from "#/i18n/declaration";
import { useDeleteOrganization } from "#/hooks/mutation/use-delete-organization";
import { useOrganization } from "#/hooks/query/use-organization";

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
    <ModalBackdrop onClose={onClose}>
      <ModalBody className="items-start border border-tertiary">
        <div className="flex flex-col gap-2">
          <BaseModalTitle title={t(I18nKey.ORG$DELETE_ORGANIZATION)} />
          <BaseModalDescription>{confirmationMessage}</BaseModalDescription>
        </div>
        <div
          className="flex flex-col gap-2 w-full"
          onClick={(event) => event.stopPropagation()}
        >
          <BrandButton
            type="button"
            variant="primary"
            onClick={handleConfirm}
            className="w-full flex items-center justify-center"
            isDisabled={isPending}
            data-testid="delete-org-confirmation"
          >
            {isPending ? (
              <LoadingSpinner size="small" />
            ) : (
              t(I18nKey.ACTION$CONFIRM_DELETE)
            )}
          </BrandButton>
          <BrandButton
            type="button"
            variant="secondary"
            onClick={onClose}
            className="w-full"
            isDisabled={isPending}
            data-testid="cancel-button"
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
