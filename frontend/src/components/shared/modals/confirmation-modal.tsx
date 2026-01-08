import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";

interface ConfirmationModalProps {
  text: string;
  onConfirm: () => void;
  onClose: () => void;
}

export function ConfirmationModal({
  text,
  onConfirm,
  onClose,
}: ConfirmationModalProps) {
  const { t } = useTranslation();
  return (
    <div
      data-testid="confirmation-modal"
      className="bg-base-secondary p-4 rounded-xl flex flex-col gap-4 border border-tertiary"
    >
      <p>{text}</p>
      <div className="w-full flex gap-2">
        <BrandButton
          testId="cancel-button"
          type="button"
          onClick={onClose}
          variant="secondary"
          className="grow"
        >
          {t(I18nKey.BUTTON$CANCEL)}
        </BrandButton>
        <BrandButton
          testId="confirm-button"
          type="button"
          onClick={onConfirm}
          variant="primary"
          className="grow"
        >
          {t(I18nKey.BUTTON$CONFIRM)}
        </BrandButton>
      </div>
    </div>
  );
}
