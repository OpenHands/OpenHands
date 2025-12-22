import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import OpenHandsLogo from "#/assets/branding/openhands-logo.svg?react";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { TermsAndPrivacyNotice } from "#/components/shared/terms-and-privacy-notice";

interface EmailVerificationModalProps {
  onClose: () => void;
}

export function EmailVerificationModal({
  onClose,
}: EmailVerificationModalProps) {
  const { t } = useTranslation();

  return (
    <ModalBackdrop onClose={onClose}>
      <ModalBody className="border border-tertiary">
        <OpenHandsLogo width={68} height={46} />
        <div className="flex flex-col gap-2 w-full items-center text-center">
          <h1 className="text-2xl font-bold">
            {t(I18nKey.AUTH$PLEASE_CHECK_EMAIL_TO_VERIFY)}
          </h1>
        </div>

        <TermsAndPrivacyNotice />
      </ModalBody>
    </ModalBackdrop>
  );
}
