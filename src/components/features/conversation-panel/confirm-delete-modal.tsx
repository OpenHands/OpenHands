import React from "react";
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
  title?: string;
  description?: React.ReactNode;
}

export function ConfirmDeleteModal({
  onConfirm,
  onCancel,
  conversationTitle,
  title,
  description,
}: ConfirmDeleteModalProps) {
  const { t } = useTranslation("openhands");

  let confirmationMessage: React.ReactNode;
  if (description != null) {
    confirmationMessage = description;
  } else if (conversationTitle) {
    confirmationMessage = (
      <Trans
        ns="openhands"
        i18nKey={I18nKey.CONVERSATION$DELETE_WARNING_WITH_TITLE}
        values={{ title: conversationTitle }}
        components={{ title: <span className="text-white" /> }}
      />
    );
  } else {
    confirmationMessage = t(I18nKey.CONVERSATION$DELETE_WARNING);
  }

  return (
    <ModalBackdrop onClose={onCancel}>
      <ModalBody className="items-start border border-[var(--oh-border)]">
        <div className="flex flex-col gap-2">
          <BaseModalTitle
            title={title ?? t(I18nKey.CONVERSATION$CONFIRM_DELETE)}
          />
          <BaseModalDescription>{confirmationMessage}</BaseModalDescription>
        </div>
        <div
          className="flex justify-end gap-2 w-full"
          onClick={(event) => event.stopPropagation()}
        >
          <BrandButton
            type="button"
            variant="secondary"
            onClick={onCancel}
            data-testid="cancel-button"
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
          <BrandButton
            type="button"
            variant="primary"
            onClick={onConfirm}
            data-testid="confirm-button"
          >
            {t(I18nKey.ACTION$CONFIRM_DELETE)}
          </BrandButton>
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
