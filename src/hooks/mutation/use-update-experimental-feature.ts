import {
  useMutation,
  useQueryClient,
  type MutateOptions,
} from "@tanstack/react-query";
import type { ExperimentalFeature } from "@openhands/typescript-client/clients";
import { experimentalFeaturesService } from "#/api/experimental-features-service/experimental-features-service.api";
import type { Backend } from "#/api/backend-registry/types";
import { experimentalFeaturesQueryKey } from "#/hooks/query/use-experimental-features";
import { useActiveBackend } from "#/contexts/active-backend-context";

interface FeatureSwitch {
  /** Fixed when the person flips the switch, so a queued or slow request
   * still goes to, and is cached for, the backend they were looking at. */
  backend: Backend;
  key: string;
  enabled: boolean;
}

/**
 * Turn one experimental feature on or off on the active backend.
 *
 * The cache stays true to that backend's server:
 * - reads of its list still in flight are cancelled when the switch is sent,
 *   so an older answer cannot land after the new one;
 * - the server's answer goes into that backend's list;
 * - the list is then fetched again from the server to reconcile.
 */
export const useUpdateExperimentalFeature = () => {
  const queryClient = useQueryClient();
  const { backend } = useActiveBackend();
  const mutation = useMutation<ExperimentalFeature, Error, FeatureSwitch>({
    mutationFn: experimentalFeaturesService.updateExperimentalFeature,
    onMutate: ({ backend: target }) =>
      queryClient.cancelQueries({
        queryKey: experimentalFeaturesQueryKey(target),
      }),
    onSuccess: async (updated, { backend: target }) => {
      const queryKey = experimentalFeaturesQueryKey(target);
      // A read may have started while the switch was in flight.
      await queryClient.cancelQueries({ queryKey });
      queryClient.setQueryData<ExperimentalFeature[]>(queryKey, (current) =>
        current?.map((feature) =>
          feature.key === updated.key ? updated : feature,
        ),
      );
    },
    onSettled: (_data, _error, { backend: target }) =>
      queryClient.invalidateQueries({
        queryKey: experimentalFeaturesQueryKey(target),
      }),
  });
  return {
    ...mutation,
    mutate: (
      change: Omit<FeatureSwitch, "backend">,
      options?: MutateOptions<ExperimentalFeature, Error, FeatureSwitch>,
    ) => mutation.mutate({ ...change, backend }, options),
  };
};
