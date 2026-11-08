import { useMutation, useQueryClient } from "@tanstack/react-query";
import type { FeatureFlag } from "@openhands/typescript-client/clients";
import { experimentalFeaturesService } from "#/api/experimental-features-service/experimental-features-service.api";
import { EXPERIMENTAL_FEATURES_QUERY_KEY } from "#/hooks/query/use-experimental-features";
import { useActiveBackend } from "#/contexts/active-backend-context";

export const useUpdateExperimentalFeature = () => {
  const queryClient = useQueryClient();
  const backendId = useActiveBackend().backend.id;
  return useMutation({
    mutationFn: experimentalFeaturesService.updateExperimentalFeature,
    onSuccess: (updated: FeatureFlag) => {
      // Only this backend's list: other backends keep their own flags.
      queryClient.setQueryData<FeatureFlag[]>(
        [EXPERIMENTAL_FEATURES_QUERY_KEY, backendId],
        (current) =>
          current?.map((feature) =>
            feature.key === updated.key ? updated : feature,
          ),
      );
    },
  });
};
