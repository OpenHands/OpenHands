import { useMutation } from "@tanstack/react-query";
import { useNavigate } from "react-router";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

type SubmitOnboardingArgs = {
  selections: Record<string, string>;
};

export const useSubmitOnboarding = () => {
  const navigate = useNavigate();

  return useMutation({
    mutationFn: async ({ selections }: SubmitOnboardingArgs) =>
      // TODO: mark onboarding as complete
      // TODO: persist user responses
      ({ selections }),
    onSuccess: () => {
      const finalRedirectUrl = "/"; // TODO: use redirect url from api response
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
