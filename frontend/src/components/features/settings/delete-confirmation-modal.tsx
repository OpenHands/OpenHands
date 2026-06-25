import React from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { I18nKey } from "#/i18n/declaration";

interface DeleteConfirmationModalProps {
  isOpen: boolean;
  itemName: string;
  onClose: () => void;
  onDelete: () => void;
  isDeleting?: boolean;
}

export function DeleteConfirmationModal({
  isOpen,
  itemName,
  onClose,
  onDelete,
  isDeleting = false,
}: DeleteConfirmationModalProps) {
  const { t } = useTranslation();

  if (!isOpen) return null;

  return (
    <ModalBackdrop>
      <div
        className="bg-base p-6 rounded-xl flex flex-col gap-4 border border-tertiary"
        style={{ width: "400px" }}
      >
        <h3 className="text-xl font-bold">
          {t(I18nKey.SETTINGS$DELETE_CONFIRMATION_TITLE)}
        </h3>

        <p className="text-sm text-tertiary-alt">
          {t(I18nKey.SETTINGS$DELETE_CONFIRMATION_MESSAGE, {
            name: itemName,
          })}
        </p>

        <div className="w-full flex gap-2 mt-2">
          <BrandButton
            type="button"
            variant="secondary"
            className="grow"
            onClick={onClose}
            isDisabled={isDeleting}
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
          <BrandButton
            testId="confirm-delete-button"
            type="button"
            variant="danger"
            className="grow"
            onClick={onDelete}
            isDisabled={isDeleting}
          >
            {isDeleting ? <span>...</span> : t(I18nKey.BUTTON$DELETE)}
          </BrandButton>
        </div>
      </div>
    </ModalBackdrop>
  );
}
