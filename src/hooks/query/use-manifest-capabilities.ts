import { useQuery } from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  assessCapabilityRequirements,
  type SetupCapabilitySupport,
} from "#/manifests/manifest-capabilities";
import type { DeploymentCapabilities, SetupEntry } from "#/manifests/types";
import { SETUP_QUERY_KEYS } from "./query-keys";

export interface SetupCapabilitiesResult {
  /** The discovery response, or null when there is nothing to report. */
  capabilities: DeploymentCapabilities | null;
  supported: SetupCapabilitySupport;
  /** Which requirements a `false` verdict came from; empty otherwise. */
  unmet: string[];
  isLoading: boolean;
}

/** Fetch the deployment-owned automation limits and feature set. */
export function useDeploymentCapabilities() {
  const { backend, orgId } = useActiveBackend();

  return useQuery({
    queryKey: [...SETUP_QUERY_KEYS.capabilities(), backend.id, orgId],
    queryFn: () => AutomationService.getCapabilities(),
    retry: false,
    staleTime: 1000 * 60 * 5,
    // An older deployment without discovery is an expected compatibility state.
    meta: { disableToast: true },
  });
}

/**
 * Ask the deployment what it supports, then compare that against what the
 * manifest requires.
 *
 * A deployment that does not answer resolves to "unknown" rather than
 * unsupported. Discovery is not implemented on every deployment yet, and
 * refusing to render on a failed probe would block every manifest on the ones
 * that simply cannot be asked. Nothing is resolved into the form in that case
 * either, so the manifest's own defaults stand.
 */
export function useSetupCapabilities(
  entry: SetupEntry,
): SetupCapabilitiesResult {
  const query = useDeploymentCapabilities();

  if (query.isLoading) {
    return {
      capabilities: null,
      supported: "unknown",
      unmet: [],
      isLoading: true,
    };
  }

  if (!query.data) {
    return {
      capabilities: null,
      supported: "unknown",
      unmet: [],
      isLoading: false,
    };
  }

  const assessment = assessCapabilityRequirements(entry, query.data);

  return {
    capabilities: query.data,
    supported: assessment.supported,
    unmet: assessment.unmet,
    isLoading: false,
  };
}
