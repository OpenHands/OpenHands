import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import OpenHandsLogo from "#/assets/branding/openhands-logo.svg?react";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { TermsAndPrivacyNotice } from "#/components/shared/terms-and-privacy-notice";
import { BrandButton } from "../settings/brand-button";
import { useEmailVerification } from "#/hooks/use-email-verification";

interface EmailVerificationModalProps {
  onClose: () => void;
  userId?: string | null;
}

export function EmailVerificationModal({
  onClose,
  userId,
}: EmailVerificationModalProps) {
  const { t } = useTranslation();
  const {
    resendEmailVerification,
    isResendingVerification: isResending,
    isCooldownActive,
    formattedCooldownTime,
  } = useEmailVerification();

  let resendButtonLabel: string;
  if (isResending) {
    resendButtonLabel = t(I18nKey.SETTINGS$SENDING);
  } else if (isCooldownActive) {
    resendButtonLabel = `${t(I18nKey.SETTINGS$RESEND_VERIFICATION)} (${formattedCooldownTime})`;
  } else {
    resendButtonLabel = t(I18nKey.SETTINGS$RESEND_VERIFICATION);
  }

  return (
    <ModalBackdrop onClose={onClose}>
      <ModalBody className="border border-tertiary">
        <OpenHandsLogo width={68} height={46} />
        <div className="flex flex-col gap-2 w-full items-center text-center">
          <h1 className="text-2xl font-bold">
            {t(I18nKey.AUTH$PLEASE_CHECK_EMAIL_TO_VERIFY)}
          </h1>
        </div>

        <div className="flex flex-col gap-3 w-full mt-4">
          <BrandButton
            type="button"
            variant="primary"
            onClick={() =>
              resendEmailVerification({ userId, isAuthFlow: true })
            }
            isDisabled={isResending || isCooldownActive}
            className="w-full font-semibold"
          >
            {resendButtonLabel}
          </BrandButton>
        </div>

        <TermsAndPrivacyNotice />
      </ModalBody>
    </ModalBackdrop>
  );
}
