import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import {
  BaseModalTitle,
  BaseModalDescription,
} from "#/components/shared/modals/confirmation-modals/base-modal";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { FeedbackForm } from "./feedback-form";

interface FeedbackModalProps {
  onClose: () => void;
  polarity: "positive" | "negative";
}

export function FeedbackModal({ onClose, polarity }: FeedbackModalProps) {
  const { t } = useTranslation();

  return (
    <ModalBody className="border border-tertiary">
      <BaseModalTitle title={t(I18nKey.FEEDBACK$TITLE)} />
      <BaseModalDescription description={t(I18nKey.FEEDBACK$DESCRIPTION)} />
      <FeedbackForm onClose={onClose} polarity={polarity} />
    </ModalBody>
  );
}
