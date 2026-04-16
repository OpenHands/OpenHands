import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router";
import { openHands } from "#/api/open-hands-axios";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

type SubmitOnboardingArgs = {
  selections: Record<string, string | string[]>;
};

export const useSubmitOnboarding = () => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ selections }: SubmitOnboardingArgs) => {
      // Mark onboarding as complete
      await openHands.post("/api/complete_onboarding");
      return { selections };
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["settings"] });

      const finalRedirectUrl = "/";
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
    onError: (error) => {
      displayErrorToast(error.message);
      window.location.href = "/";
    },
  });
};
