import { useQuery } from "@tanstack/react-query";
import { onboardingService } from "#/api/onboarding-service/onboarding-service.api";
import { useConfig } from "./use-config";
import { useIsAuthed } from "./use-is-authed";

export const useOnboardingStatus = () => {
  const { data: config } = useConfig();
  const { data: isAuthed } = useIsAuthed();

  return useQuery({
    queryKey: ["onboarding-status"],
    queryFn: onboardingService.getStatus,
    enabled: config?.app_mode === "saas" && !!isAuthed,
    staleTime: 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
    retry: false,
    meta: {
      disableToast: true,
    },
  });
};
