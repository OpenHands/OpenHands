import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { Typography } from "#/ui/typography";
import { cn } from "#/utils/utils";
import CloseIcon from "#/icons/modal-close.svg?react";

interface RequestSubmittedModalProps {
  onClose: () => void;
}

export function RequestSubmittedModal({ onClose }: RequestSubmittedModalProps) {
  const { t } = useTranslation();

  return (
    <ModalBackdrop
      onClose={onClose}
      aria-label={t(I18nKey.ENTERPRISE$REQUEST_SUBMITTED_TITLE)}
    >
      <div
        data-testid="request-submitted-modal"
        className={cn(
          "w-[448px] bg-black rounded-md shadow-lg",
          "border border-[#242424] border-t-[#242424]",
        )}
      >
        {/* Header with close button */}
        <div className={cn("relative p-6 pb-0")}>
          <button
            type="button"
            onClick={onClose}
            aria-label={t(I18nKey.MODAL$CLOSE_BUTTON_LABEL)}
            className={cn(
              "absolute top-[17px] right-[17px] w-4 h-4",
              "flex items-center justify-center rounded-sm cursor-pointer",
              "opacity-70 hover:opacity-100 transition-opacity",
            )}
          >
            <CloseIcon className={cn("w-4 h-4")} />
          </button>

          {/* Title and description */}
          <div className={cn("flex flex-col gap-1.5 pr-8")}>
            <Typography.H2
              className={cn("text-lg leading-[18px] tracking-[-0.45px]")}
            >
              {t(I18nKey.ENTERPRISE$REQUEST_SUBMITTED_TITLE)}
            </Typography.H2>
            <Typography.Text className={cn("text-[#8C8C8C] leading-5")}>
              {t(I18nKey.ENTERPRISE$REQUEST_SUBMITTED_DESCRIPTION)}
            </Typography.Text>
          </div>
        </div>

        {/* Footer with Done button */}
        <div className={cn("p-6 pt-4 flex justify-end")}>
          <button
            type="button"
            onClick={onClose}
            aria-label={t(I18nKey.ENTERPRISE$DONE_BUTTON)}
            className={cn(
              "px-4 py-2 text-sm font-medium rounded cursor-pointer",
              "bg-white text-black hover:bg-gray-100 transition-colors",
            )}
          >
            {t(I18nKey.ENTERPRISE$DONE_BUTTON)}
          </button>
        </div>
      </div>
    </ModalBackdrop>
  );
}
