import React from "react";
import { useTranslation } from "react-i18next";
import { useSearchParams } from "react-router";
import { I18nKey } from "#/i18n/declaration";
import OpenHandsLogo from "#/assets/branding/openhands-logo.svg?react";
import { TOSCheckbox } from "#/components/features/waitlist/tos-checkbox";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { useAcceptTos } from "#/hooks/mutation/use-accept-tos";

export default function AcceptTOS() {
  const { t } = useTranslation();
  const [searchParams] = useSearchParams();
  const [isTosAccepted, setIsTosAccepted] = React.useState(false);

  // Get the redirect URL from the query parameters
  const redirectUrl = searchParams.get("redirect_url") || "/";

  // Use mutation for accepting TOS
  const { mutate: acceptTOS, isPending: isSubmitting } = useAcceptTos();

  const handleAcceptTOS = () => {
    if (isTosAccepted && !isSubmitting) {
      acceptTOS({ redirectUrl });
    }
  };

  return (
    <ModalBackdrop>
      <div className="border border-tertiary p-8 rounded-lg max-w-md w-full flex flex-col gap-6 items-center bg-base-secondary">
        <OpenHandsLogo width={68} height={46} />

        <div className="flex flex-col gap-2 w-full items-center text-center">
          <h1 className="text-2xl font-bold">
            {t(I18nKey.TOS$ACCEPT_TERMS_OF_SERVICE)}
          </h1>
          <p className="text-sm text-gray-500">
            {t(I18nKey.TOS$ACCEPT_TERMS_DESCRIPTION)}
          </p>
        </div>

        <TOSCheckbox onChange={() => setIsTosAccepted((prev) => !prev)} />

        <BrandButton
          isDisabled={!isTosAccepted || isSubmitting}
          type="button"
          variant="primary"
          onClick={handleAcceptTOS}
          className="w-full font-semibold"
        >
          {isSubmitting ? t(I18nKey.HOME$LOADING) : t(I18nKey.TOS$CONTINUE)}
        </BrandButton>
      </div>
    </ModalBackdrop>
  );
}
