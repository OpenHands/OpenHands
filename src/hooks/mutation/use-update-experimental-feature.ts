import { useMutation, useQueryClient } from "@tanstack/react-query";
import type { ExperimentalFeature } from "@openhands/typescript-client/clients";
import { experimentalFeaturesService } from "#/api/experimental-features-service/experimental-features-service.api";
import { EXPERIMENTAL_FEATURES_QUERY_KEY } from "#/hooks/query/use-experimental-features";
import { useActiveBackend } from "#/contexts/active-backend-context";

export const useUpdateExperimentalFeature = () => {
  const queryClient = useQueryClient();
  const backendId = useActiveBackend().backend.id;
  return useMutation<
    ExperimentalFeature,
    Error,
    { key: string; enabled: boolean },
    { backendId: string }
  >({
    mutationFn: experimentalFeaturesService.updateExperimentalFeature,
    // The backend the switch was sent to: the person may switch backends
    // before it answers, and the answer belongs to this one's list only.
    onMutate: () => ({ backendId }),
    onSuccess: (updated, _variables, context) => {
      queryClient.setQueryData<ExperimentalFeature[]>(
        [EXPERIMENTAL_FEATURES_QUERY_KEY, context.backendId],
        (current) =>
          current?.map((feature) =>
            feature.key === updated.key ? updated : feature,
          ),
      );
    },
  });
};
