import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router";
import { openHands } from "#/api/open-hands-axios";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { navigateOrHardRedirect } from "#/utils/cross-app-redirect";

type SubmitOnboardingArgs = {
  selections: Record<string, string | string[]>;
  /**
   * Fallback destination to navigate to when the server response does
   * not include a ``redirect_url``. ``OnboardingForm`` passes the
   * caller's ``returnTo`` here so deep links survive the onboarding
   * interstitial. Defaults to ``/`` when omitted.
   */
  returnTo?: string;
};

interface OnboardingResponse {
  status: string;
  redirect_url: string;
}

export const useSubmitOnboarding = () => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ selections }: SubmitOnboardingArgs) => {
      const { data } = await openHands.post<OnboardingResponse>(
        "/api/complete_onboarding",
        { selections },
      );
      return data;
    },
    onSuccess: (data, { returnTo }) => {
      queryClient.invalidateQueries({ queryKey: SETTINGS_QUERY_KEYS.all });
      queryClient.invalidateQueries({ queryKey: ["onboarding-status"] });

      const finalRedirectUrl = data.redirect_url || returnTo || "/";
      // Check if the redirect URL is an external URL (starts with http or https)
      if (
        finalRedirectUrl.startsWith("http://") ||
        finalRedirectUrl.startsWith("https://")
      ) {
        // For external URLs, redirect using window.location
        window.location.href = finalRedirectUrl;
      } else {
        navigateOrHardRedirect(navigate, finalRedirectUrl);
      }
    },
    onError: (error) => {
      displayErrorToast(error.message);
      window.location.href = "/";
    },
  });
};
