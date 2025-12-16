import { useMutation } from "@tanstack/react-query";
import { usePostHog } from "posthog-js/react";
import { useNavigate } from "react-router";
import { openHands } from "#/api/open-hands-axios";
import { handleCaptureConsent } from "#/utils/handle-capture-consent";
import { useTracking } from "#/hooks/use-tracking";

interface AcceptTosVariables {
  redirectUrl: string;
}

interface AcceptTosResponse {
  redirect_url?: string;
}

export const useAcceptTos = () => {
  const posthog = usePostHog();
  const navigate = useNavigate();
  const { trackUserSignupCompleted } = useTracking();

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
};
