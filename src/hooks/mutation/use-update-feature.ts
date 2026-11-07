import { useMutation, useQueryClient } from "@tanstack/react-query";
import type { FeatureFlag } from "@openhands/typescript-client/clients";
import { featuresService } from "#/api/features-service/features-service.api";
import { FEATURES_QUERY_KEY } from "#/hooks/query/use-features";
import { useActiveBackend } from "#/contexts/active-backend-context";

export const useUpdateFeature = () => {
  const queryClient = useQueryClient();
  const backendId = useActiveBackend().backend.id;
  return useMutation({
    mutationFn: featuresService.updateFeature,
    onSuccess: (updated: FeatureFlag) => {
      // Only this backend's list: other backends keep their own flags.
      queryClient.setQueryData<FeatureFlag[]>(
        [FEATURES_QUERY_KEY, backendId],
        (current) =>
          current?.map((feature) =>
            feature.key === updated.key ? updated : feature,
          ),
      );
    },
  });
};
