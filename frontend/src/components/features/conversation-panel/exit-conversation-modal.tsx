import { useTranslation } from "react-i18next";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { ModalButton } from "#/components/shared/buttons/modal-button";
import { BaseModalTitle } from "#/components/shared/modals/confirmation-modals/base-modal";
import { I18nKey } from "#/i18n/declaration";

interface ExitConversationModalProps {
  onConfirm: () => void;
  onClose: () => void;
}

export function ExitConversationModal({
  onConfirm,
  onClose,
}: ExitConversationModalProps) {
  const { t } = useTranslation();

  return (
    <ModalBody testID="confirm-new-conversation-modal">
      <BaseModalTitle title={t(I18nKey.CONVERSATION$EXIT_WARNING)} />
      <div className="flex w-full gap-2">
        <ModalButton
          text={t(I18nKey.ACTION$CONFIRM)}
          onClick={onConfirm}
          className="bg-[#C63143] flex-1"
        />
        <ModalButton
          text={t(I18nKey.BUTTON$CANCEL)}
          onClick={onClose}
          className="bg-tertiary flex-1"
        />
      </div>
    </ModalBody>
  );
}
