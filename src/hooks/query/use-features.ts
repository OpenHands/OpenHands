import { useQuery } from "@tanstack/react-query";
import type { FeatureFlag } from "@openhands/typescript-client/clients";
import { featuresService } from "#/api/features-service/features-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";

export const FEATURES_QUERY_KEY = "settings-features";

/** The user's feature flags; an older agent server without them lists none. */
export const useFeatures = () => {
  const active = useActiveBackend();
  return useQuery<FeatureFlag[], Error>({
    queryKey: [FEATURES_QUERY_KEY, active.backend.id],
    queryFn: async () => {
      try {
        return await featuresService.listFeatures();
      } catch {
        return [];
      }
    },
    staleTime: 1000 * 60,
  });
};

export const useIsFeatureEnabled = (key: string): boolean => {
  const { data } = useFeatures();
  return (
    data?.some((feature) => feature.key === key && feature.enabled) ?? false
  );
};
