import { useQuery } from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  evaluateCapabilityRequirements,
  type ManifestCapabilitySupport,
} from "#/manifests/manifest-capabilities";
import type { ExtensionManifest } from "#/manifests/types";
import { MANIFEST_QUERY_KEYS } from "./query-keys";

export interface ManifestCapabilitiesResult {
  /** The discovery response, or null when there is nothing to report. */
  capabilities: Record<string, unknown> | null;
  supported: ManifestCapabilitySupport;
  isLoading: boolean;
}

/**
 * Ask the deployment what it supports, then compare that against what the
 * manifest requires.
 *
 * A deployment that does not answer resolves to "unknown" rather than
 * unsupported. Discovery is not implemented everywhere yet, and refusing to
 * render on a failed probe would block every manifest on deployments that
 * simply cannot be asked. Nothing is bound into the form in that case either,
 * so the manifest's own defaults stand.
 */
export function useManifestCapabilities(
  manifest: ExtensionManifest,
): ManifestCapabilitiesResult {
  const discovery = manifest.capabilities?.discovery;
  const { backend, orgId } = useActiveBackend();

  const query = useQuery({
    queryKey: [
      ...MANIFEST_QUERY_KEYS.capabilities(discovery?.path ?? ""),
      backend.id,
      orgId,
    ],
    queryFn: () => AutomationService.getCapabilities(discovery!.path),
    enabled: !!discovery,
    retry: false,
    staleTime: 1000 * 60 * 5,
    // A deployment without discovery is an expected state, not a user-facing
    // failure.
    meta: { disableToast: true },
  });

  const requires = manifest.capabilities?.requires;
  if (!requires) {
    return { capabilities: null, supported: true, isLoading: false };
  }

  if (query.isLoading) {
    return { capabilities: null, supported: "unknown", isLoading: true };
  }

  if (!query.data) {
    return { capabilities: null, supported: "unknown", isLoading: false };
  }

  return {
    capabilities: query.data,
    supported: evaluateCapabilityRequirements(requires, query.data),
    isLoading: false,
  };
}
