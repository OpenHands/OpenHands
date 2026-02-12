import { Trans, useTranslation } from "react-i18next";
import {
  BaseModalDescription,
  BaseModalTitle,
} from "#/components/shared/modals/confirmation-modals/base-modal";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { BrandButton } from "../settings/brand-button";
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
    <ModalBackdrop onClose={onCancel}>
      <ModalBody className="items-start border border-tertiary">
        <div className="flex flex-col gap-2">
          <BaseModalTitle title={t(I18nKey.ORG$CONFIRM_REMOVE_MEMBER)} />
          <BaseModalDescription>{confirmationMessage}</BaseModalDescription>
        </div>
        <div
          className="flex flex-col gap-2 w-full"
          onClick={(event) => event.stopPropagation()}
        >
          <BrandButton
            type="button"
            variant="primary"
            onClick={onConfirm}
            className="w-full flex items-center justify-center"
            testId="confirm-button"
            isDisabled={isLoading}
          >
            {isLoading ? (
              <LoadingSpinner size="small" />
            ) : (
              t(I18nKey.ACTION$CONFIRM_DELETE)
            )}
          </BrandButton>
          <BrandButton
            type="button"
            variant="secondary"
            onClick={onCancel}
            className="w-full"
            testId="cancel-button"
            isDisabled={isLoading}
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
