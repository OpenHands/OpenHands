import { useQuery } from "@tanstack/react-query";
import { HttpError } from "@openhands/typescript-client";
import type { FeatureFlag } from "@openhands/typescript-client/clients";
import { experimentalFeaturesService } from "#/api/experimental-features-service/experimental-features-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";

export const EXPERIMENTAL_FEATURES_QUERY_KEY = "settings-experimental";

/** The user's feature flags; an older agent server without them lists none. */
export const useExperimentalFeatures = () => {
  const active = useActiveBackend();
  return useQuery<FeatureFlag[], Error>({
    queryKey: [EXPERIMENTAL_FEATURES_QUERY_KEY, active.backend.id],
    queryFn: async () => {
      try {
        return await experimentalFeaturesService.listExperimentalFeatures();
      } catch (error) {
        // An agent server older than feature flags has no such route.
        if (error instanceof HttpError && error.status === 404) return [];
        throw error;
      }
    },
    staleTime: 1000 * 60,
  });
};

export const useIsExperimentalFeatureEnabled = (key: string): boolean => {
  const { data } = useExperimentalFeatures();
  return (
    data?.some((feature) => feature.key === key && feature.enabled) ?? false
  );
};
