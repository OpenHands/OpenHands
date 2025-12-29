import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

interface TermsAndPrivacyNoticeProps {
  className?: string;
}

export function TermsAndPrivacyNotice({
  className = "mt-4 text-xs text-center text-muted-foreground",
}: TermsAndPrivacyNoticeProps) {
  const { t } = useTranslation();

  return (
    <p className={className} data-testid="terms-and-privacy-notice">
      {t(I18nKey.AUTH$BY_SIGNING_UP_YOU_AGREE_TO_OUR)}{" "}
      <a
        href="https://www.all-hands.dev/tos"
        target="_blank"
        className="underline hover:text-primary"
        rel="noopener noreferrer"
      >
        {t(I18nKey.COMMON$TERMS_OF_SERVICE)}
      </a>{" "}
      {t(I18nKey.COMMON$AND)}{" "}
      <a
        href="https://www.all-hands.dev/privacy"
        target="_blank"
        className="underline hover:text-primary"
        rel="noopener noreferrer"
      >
        {t(I18nKey.COMMON$PRIVACY_POLICY)}
      </a>
      .
    </p>
  );
}
