import React from "react";
import { useTranslation } from "react-i18next";
import { useNavigate, useSearchParams } from "react-router";
import { useMutation } from "@tanstack/react-query";
import { usePostHog } from "posthog-js/react";
import { I18nKey } from "#/i18n/declaration";
import OpenHandsLogo from "#/assets/branding/openhands-logo.svg?react";
import { TOSCheckbox } from "#/components/features/waitlist/tos-checkbox";
import { BrandButton } from "#/components/features/settings/brand-button";
import { handleCaptureConsent } from "#/utils/handle-capture-consent";
import { openHands } from "#/api/open-hands-axios";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { useTracking } from "#/hooks/use-tracking";

export default function AcceptTOS() {
  const posthog = usePostHog();
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [isTosAccepted, setIsTosAccepted] = React.useState(false);
  const { trackUserSignupCompleted } = useTracking();

  // Get the redirect URL from the query parameters
  const redirectUrl = searchParams.get("redirect_url") || "/";

  // Use mutation for accepting TOS
  const { mutate: acceptTOS, isPending: isSubmitting } = useMutation({
    mutationFn: async () => {
      // Set consent for analytics
      handleCaptureConsent(posthog, true);

      // Call the API to record TOS acceptance in the database
      return openHands.post("/api/accept_tos", {
        redirect_url: redirectUrl,
      });
    },
    onSuccess: (response) => {
      // Track user signup completion
      trackUserSignupCompleted();

      // Get the redirect URL from the response
      const finalRedirectUrl = response.data.redirect_url || redirectUrl;

      // Check if the redirect URL is an external URL (starts with http or https)
      if (
        finalRedirectUrl.startsWith("http://") ||
        finalRedirectUrl.startsWith("https://")
      ) {
        // For external URLs, redirect using window.location
        window.location.href = finalRedirectUrl;
      } else {
        // For internal routes, use navigate
        navigate(finalRedirectUrl);
      }
    },
    onError: () => {
      window.location.href = "/";
    },
  });

  const handleAcceptTOS = () => {
    if (isTosAccepted && !isSubmitting) {
      acceptTOS();
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
