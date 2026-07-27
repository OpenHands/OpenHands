import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ModalBackdrop } from "./modal-backdrop";

interface ConfirmationModalProps {
  text: string;
  onConfirm: () => void;
  onCancel: () => void;
  /**
   * Disables both action buttons while an asynchronous confirm
   * mutation is in flight. Defaults to false to preserve existing
   * call sites that don't track mutation state.
   */
  isConfirming?: boolean;
}

export function ConfirmationModal({
  text,
  onConfirm,
  onCancel,
  isConfirming = false,
}: ConfirmationModalProps) {
  const { t } = useTranslation("openhands");
  // Suppress the backdrop's click / Escape close handler while the
  // confirm mutation is in flight; otherwise the user could dismiss
  // the modal mid-request and never see the result (the buttons are
  // already disabled, but the backdrop wasn't).
  return (
    <ModalBackdrop
      onClose={isConfirming ? undefined : onCancel}
      closeOnEscape={!isConfirming}
    >
      <div
        data-testid="confirmation-modal"
        className="bg-base-secondary p-4 rounded-xl flex flex-col gap-4 border border-[var(--oh-border)]"
      >
        <p>{text}</p>
        <div className="w-full flex justify-end gap-2">
          <BrandButton
            testId="cancel-button"
            type="button"
            onClick={onCancel}
            variant="secondary"
            isDisabled={isConfirming}
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
          <BrandButton
            testId="confirm-button"
            type="button"
            onClick={onConfirm}
            variant="primary"
            isDisabled={isConfirming}
          >
            {t(I18nKey.BUTTON$CONFIRM)}
          </BrandButton>
        </div>
      </div>
    </ModalBackdrop>
  );
}
