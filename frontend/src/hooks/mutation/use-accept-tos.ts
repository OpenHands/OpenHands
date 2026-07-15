import { useMutation } from "@tanstack/react-query";
import { usePostHog } from "posthog-js/react";
import { useNavigate } from "react-router";
import { openHands } from "#/api/open-hands-axios";
import {
  isCrossOriginUrl,
  navigateToReturnUrl,
} from "#/utils/canvas-return-url";
import { handleCaptureConsent } from "#/utils/handle-capture-consent";

interface AcceptTosVariables {
  redirectUrl: string;
}

interface AcceptTosResponse {
  redirect_url?: string;
}

export const useAcceptTos = () => {
  const posthog = usePostHog();
  const navigate = useNavigate();

  return useMutation({
    mutationFn: async ({ redirectUrl }: AcceptTosVariables) => {
      // Set consent for analytics
      handleCaptureConsent(posthog, true);

      // Call the API to record TOS acceptance in the database
      return openHands.post<AcceptTosResponse>("/api/accept_tos", {
        redirect_url: redirectUrl,
      });
    },
    onSuccess: (response, { redirectUrl }) => {
      // Get the redirect URL from the response
      const finalRedirectUrl = response.data.redirect_url || redirectUrl;

      if (isCrossOriginUrl(finalRedirectUrl)) {
        window.location.href = finalRedirectUrl;
        return;
      }

      navigateToReturnUrl(finalRedirectUrl, navigate);
    },
    onError: () => {
      window.location.href = "/";
    },
  });
};
