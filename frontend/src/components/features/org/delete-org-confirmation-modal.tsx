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
    <ModalBackdrop
      onClose={isPending ? undefined : onClose}
      aria-label={t(I18nKey.ORG$DELETE_ORGANIZATION)}
    >
      <ModalBody
        className="items-start rounded-xl p-6 w-sm items-start flex flex-col gap-6 bg-[#171717] modal-box-shadow"
        testID="delete-org-confirmation"
      >
        <div className="flex flex-col gap-2">
          <BaseModalTitle title={t(I18nKey.ORG$DELETE_ORGANIZATION)} />
          <BaseModalDescription>{confirmationMessage}</BaseModalDescription>
        </div>
        <div className="flex gap-2 w-full">
          <BrandButton
            type="button"
            variant="primary"
            onClick={handleConfirm}
            className="w-full flex items-center justify-center bg-[#F3CE49] text-sm leading-4 font-medium rounded h-10"
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
              t(I18nKey.BUTTON$CONFIRM)
            )}
          </BrandButton>
          <BrandButton
            type="button"
            variant="secondary"
            onClick={onClose}
            className="w-full bg-[#737373] text-sm text-white leading-4 font-medium rounded border-none h-10"
            isDisabled={isPending}
            data-testid="cancel-button"
          >
            {t(I18nKey.BUTTON$CLOSE)}
          </BrandButton>
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
