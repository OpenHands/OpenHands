import { useMutation, useQueryClient } from "@tanstack/react-query";
import type { FeatureFlag } from "@openhands/typescript-client/clients";
import { featuresService } from "#/api/features-service/features-service.api";
import { FEATURES_QUERY_KEY } from "#/hooks/query/use-features";

export const useUpdateFeature = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: featuresService.updateFeature,
    onSuccess: (updated: FeatureFlag) => {
      queryClient.setQueriesData<FeatureFlag[]>(
        { queryKey: [FEATURES_QUERY_KEY] },
        (current) =>
          current?.map((feature) =>
            feature.key === updated.key ? updated : feature,
          ),
      );
    },
  });
};
