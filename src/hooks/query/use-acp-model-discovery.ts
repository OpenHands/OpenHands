import { useQuery } from "@tanstack/react-query";
import type { ACPModelDiscovery } from "@openhands/typescript-client";
import AcpService from "#/api/acp-service/acp-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useSearchSecrets } from "#/hooks/query/use-get-secrets";
import { ACP_MODEL_DISCOVERY_QUERY_KEYS } from "#/hooks/query/query-keys";
import {
  toAcpModelOptions,
  type ACPModelOption,
} from "#/constants/acp-providers";

async function discover(
  providerKey: string,
  secretNames: string[],
): Promise<ACPModelDiscovery | null> {
  try {
    return await AcpService.discoverModels(providerKey, secretNames);
  } catch {
    // Older agent-servers lack the endpoint; callers keep the curated list.
    return null;
  }
}

/**
 * The models a built-in ACP provider offers on the active backend, and the one
 * it uses by default, as its own server reports them for the saved
 * credentials. Local backends only: elsewhere ``models`` is empty and callers
 * keep the registry's curated list.
 */
export function useAcpModelDiscovery(
  providerKey: string | null | undefined,
  { enabled = true }: { enabled?: boolean } = {},
) {
  const active = useActiveBackend();
  const isLocal = active.backend.kind === "local";
  const secrets = useSearchSecrets({ enabled: enabled && isLocal });
  const secretNames = secrets.data.map(({ name }) => name).sort();
  const queryEnabled =
    enabled && isLocal && !!providerKey && !secrets.isLoading;

  const query = useQuery<ACPModelDiscovery | null, Error>({
    queryKey: [
      ...ACP_MODEL_DISCOVERY_QUERY_KEYS.all,
      active.backend.id,
      providerKey,
      secretNames,
    ],
    queryFn: () => discover(providerKey as string, secretNames),
    enabled: queryEnabled,
    staleTime: 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
    retry: false,
    refetchOnWindowFocus: false,
  });

  const discovery = query.data ?? null;
  const models: ACPModelOption[] = toAcpModelOptions(
    discovery?.available_models,
  );
  return {
    discovery,
    models,
    defaultModelId: discovery?.current_model_id ?? null,
    isDiscovering: queryEnabled && query.isFetching && !query.data,
  };
}
