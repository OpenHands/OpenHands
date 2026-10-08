import { useEffect, useMemo } from "react";
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
import {
  readRememberedAcpModels,
  rememberAcpModels,
} from "#/utils/remembered-acp-models";

async function discover(
  providerKey: string,
  secretNames: string[],
): Promise<ACPModelDiscovery | null> {
  try {
    return await AcpService.discoverModels(providerKey, secretNames);
  } catch {
    // Older agent-servers lack the endpoint.
    return null;
  }
}

/** Where {@link useAcpModelDiscovery}'s ``models`` came from. */
export type AcpModelSource = "live" | "remembered" | "none";

/**
 * The models a built-in ACP provider offers on the active backend, and the one
 * it uses by default, as its own server reports them for the saved
 * credentials. Only local backends can ask before a conversation starts;
 * otherwise ``models`` is the list the agent last reported in a conversation.
 */
export function useAcpModelDiscovery(
  providerKey: string | null | undefined,
  { enabled = true }: { enabled?: boolean } = {},
) {
  const { backend } = useActiveBackend();
  const isLocal = backend.kind === "local";
  const secrets = useSearchSecrets({ enabled: enabled && isLocal });
  const secretNames = secrets.data.map(({ name }) => name).sort();
  const queryEnabled =
    enabled && isLocal && !!providerKey && !secrets.isLoading;

  const query = useQuery<ACPModelDiscovery | null, Error>({
    queryKey: [
      ...ACP_MODEL_DISCOVERY_QUERY_KEYS.all,
      backend.id,
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
  const liveModels = useMemo(
    () => toAcpModelOptions(discovery?.available_models),
    [discovery],
  );
  useEffect(() => {
    if (providerKey) rememberAcpModels(backend.id, providerKey, liveModels);
  }, [backend.id, providerKey, liveModels]);

  const remembered = useMemo(
    () => (enabled ? readRememberedAcpModels(backend.id, providerKey) : []),
    [enabled, backend.id, providerKey],
  );
  let models: ACPModelOption[] = [];
  let source: AcpModelSource = "none";
  if (liveModels.length) {
    models = liveModels;
    source = "live";
  } else if (remembered.length) {
    models = remembered;
    source = "remembered";
  }

  return {
    discovery,
    models,
    source,
    defaultModelId: discovery?.current_model_id ?? null,
    isDiscovering: queryEnabled && query.isFetching && !query.data,
  };
}
