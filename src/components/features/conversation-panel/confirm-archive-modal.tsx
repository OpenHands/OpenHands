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

interface ConfirmArchiveModalProps {
  onConfirm: () => void;
  onCancel: () => void;
  conversationTitle?: string;
}

export function ConfirmArchiveModal({
  onConfirm,
  onCancel,
  conversationTitle,
}: ConfirmArchiveModalProps) {
  const { t } = useTranslation("openhands");

  const confirmationMessage =
    conversationTitle != null && conversationTitle !== "" ? (
      <Trans
        ns="openhands"
        i18nKey={I18nKey.CONVERSATION$ARCHIVE_WARNING_WITH_TITLE}
        values={{ title: conversationTitle }}
        components={{ title: <span className="text-white" /> }}
      />
    ) : (
      t(I18nKey.CONVERSATION$ARCHIVE_WARNING)
    );

  return (
    <ModalBackdrop onClose={onCancel}>
      <ModalBody className="items-start border border-[var(--oh-border)]">
        <div className="flex flex-col gap-2">
          <BaseModalTitle title={t(I18nKey.CONVERSATION$CONFIRM_ARCHIVE)} />
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
            testId="cancel-button"
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
          <BrandButton
            type="button"
            variant="primary"
            onClick={onConfirm}
            testId="confirm-button"
          >
            {t(I18nKey.ACTION$CONFIRM_ARCHIVE)}
          </BrandButton>
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
