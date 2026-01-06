import { Trans, useTranslation } from "react-i18next";
import {
  BaseModalDescription,
  BaseModalTitle,
} from "#/components/shared/modals/confirmation-modals/base-modal";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { BrandButton } from "../settings/brand-button";
import { I18nKey } from "#/i18n/declaration";

interface ConfirmDeleteModalProps {
  onConfirm: () => void;
  onCancel: () => void;
  conversationTitle?: string;
}

export function ConfirmDeleteModal({
  onConfirm,
  onCancel,
  conversationTitle,
}: ConfirmDeleteModalProps) {
  const { t } = useTranslation();

  const confirmationMessage = conversationTitle ? (
    <Trans
      i18nKey={I18nKey.CONVERSATION$DELETE_WARNING_WITH_TITLE}
      values={{ title: conversationTitle }}
      components={{ title: <span className="text-white" /> }}
    />
  ) : (
    t(I18nKey.CONVERSATION$DELETE_WARNING)
  );

  return (
    <ModalBackdrop onClose={onCancel}>
      <ModalBody className="items-start border border-tertiary">
        <div className="flex flex-col gap-2">
          <BaseModalTitle title={t(I18nKey.CONVERSATION$CONFIRM_DELETE)} />
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
            className="w-full"
            data-testid="confirm-button"
          >
            {t(I18nKey.ACTION$CONFIRM_DELETE)}
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
