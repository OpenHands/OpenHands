import { useTranslation } from "react-i18next";
import ModalCloseIcon from "#/icons/modal-close.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

interface ModalCloseButtonProps {
  onClose: () => void;
  testId?: string;
  className?: string;
  disabled?: boolean;
}

export function ModalCloseButton({
  onClose,
  testId,
  className,
  disabled = false,
}: ModalCloseButtonProps) {
  const { t } = useTranslation("openhands");

  return (
    <button
      type="button"
      data-testid={testId}
      onClick={onClose}
      disabled={disabled}
      aria-label={t(I18nKey.BUTTON$CLOSE)}
      className={cn(
        "absolute right-4 top-4 z-10 flex cursor-pointer items-center justify-center rounded-sm border-0 bg-transparent p-1 text-tertiary-alt transition-colors hover:bg-surface-raised hover:text-white disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
    >
      <ModalCloseIcon aria-hidden className="size-4" />
    </button>
  );
}
