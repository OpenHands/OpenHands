import { useQuery } from "@tanstack/react-query";
import { HttpError } from "@openhands/typescript-client";
import type { ExperimentalFeature } from "@openhands/typescript-client/clients";
import { experimentalFeaturesService } from "#/api/experimental-features-service/experimental-features-service.api";
import type { Backend } from "#/api/backend-registry/types";
import { useActiveBackend } from "#/contexts/active-backend-context";

/**
 * The cache of one backend's experimental features. Each backend keeps its
 * own switches, so everything that reads or writes this cache names the
 * backend: the list is fetched from the backend in its key, and a switch's
 * answer is written to the backend it was sent to
 * (useUpdateExperimentalFeature), never to whichever backend is active when
 * the answer arrives.
 */
export const experimentalFeaturesQueryKey = (
  backend: Pick<Backend, "id" | "connectionRevision">,
) =>
  [
    "settings-experimental",
    backend.id,
    backend.connectionRevision ?? 0,
  ] as const;

/** The active backend's experimental features; an older agent server lists none. */
export const useExperimentalFeatures = () => {
  const { backend } = useActiveBackend();
  return useQuery<ExperimentalFeature[], Error>({
    queryKey: experimentalFeaturesQueryKey(backend),
    queryFn: () => listExperimentalFeatures(backend),
    staleTime: 1000 * 60,
  });
};

const listExperimentalFeatures = async (
  backend: Backend,
): Promise<ExperimentalFeature[]> => {
  try {
    return await experimentalFeaturesService.listExperimentalFeatures(backend);
  } catch (error) {
    // An agent server older than experimental features has no such route.
    if (error instanceof HttpError && error.status === 404) return [];
    throw error;
  }
};

export const useIsExperimentalFeatureEnabled = (key: string): boolean => {
  const { data } = useExperimentalFeatures();
  return (
    data?.some((feature) => feature.key === key && feature.enabled) ?? false
  );
};
