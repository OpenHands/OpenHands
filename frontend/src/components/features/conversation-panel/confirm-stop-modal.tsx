import { useTranslation } from "react-i18next";
import {
  BaseModalDescription,
  BaseModalTitle,
} from "#/components/shared/modals/confirmation-modals/base-modal";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { BrandButton } from "../settings/brand-button";
import { I18nKey } from "#/i18n/declaration";

interface ConfirmStopModalProps {
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmStopModal({
  onConfirm,
  onCancel,
}: ConfirmStopModalProps) {
  const { t } = useTranslation();

  return (
    <ModalBackdrop onClose={onCancel}>
      <ModalBody className="items-start border border-tertiary">
        <div className="flex flex-col gap-2">
          <BaseModalTitle
            title={t(I18nKey.CONVERSATION$CONFIRM_CLOSE_CONVERSATION)}
          />
          <BaseModalDescription
            description={t(I18nKey.CONVERSATION$CLOSE_CONVERSATION_WARNING)}
          />
        </div>
        <div
          className="flex flex-col gap-2 w-full"
          onClick={(event) => event.stopPropagation()}
        >
          <BrandButton
            type="button"
            variant="primary"
            onClick={onConfirm}
            className="w-full"
            data-testid="confirm-button"
          >
            {t(I18nKey.ACTION$CONFIRM_CLOSE)}
          </BrandButton>
          <BrandButton
            type="button"
            variant="secondary"
            onClick={onCancel}
            className="w-full"
            data-testid="cancel-button"
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
