import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { DEFAULT_CONCURRENT_SANDBOX_LIMIT } from "#/utils/constants";

interface ConversationLimitModalProps {
  onClose: () => void;
  limit?: number;
}

export function ConversationLimitModal({
  onClose,
  limit = DEFAULT_CONCURRENT_SANDBOX_LIMIT,
}: ConversationLimitModalProps) {
  const { t } = useTranslation();

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        data-testid="conversation-limit-modal"
        className="flex flex-col gap-6 rounded-xl border border-[#454545] bg-[#25272D] p-[30px] w-[523px] shadow-2xl"
      >
        {/* Header */}
        <div className="flex items-center gap-[17px]">
          <Typography.H3 className="text-xl font-semibold leading-6 text-white tracking-[-0.01em]">
            {t(I18nKey.CONVERSATION_LIMIT$TITLE)}
          </Typography.H3>
        </div>

        {/* Body */}
        <Typography.Paragraph className="text-sm leading-5 text-[#A3A3A3]">
          {t(I18nKey.CONVERSATION_LIMIT$DESCRIPTION, { limit })}
        </Typography.Paragraph>

        {/* Footer */}
        <div className="flex gap-6">
          <BrandButton
            type="button"
            variant="primary"
            onClick={onClose}
            className="flex-1 bg-white text-black hover:bg-gray-100"
            testId="conversation-limit-close-button"
          >
            {t(I18nKey.BUTTON$CLOSE)}
          </BrandButton>
        </div>
      </div>
    </ModalBackdrop>
  );
}
