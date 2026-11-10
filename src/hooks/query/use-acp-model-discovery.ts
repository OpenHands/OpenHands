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
import { useRememberedAcpModels } from "#/hooks/use-remembered-acp-models";
import {
  acpLaunchKey,
  acpModelScope,
  rememberAcpModels,
} from "#/utils/remembered-acp-models";

async function discover(
  providerKey: string,
  secretNames: string[],
  options?: {
    acpCommand?: readonly string[] | null;
    acpArgs?: readonly string[] | null;
  },
): Promise<ACPModelDiscovery | null> {
  try {
    const hasCommand = !!options?.acpCommand && options.acpCommand.length > 0;
    const hasArgs = !!options?.acpArgs && options.acpArgs.length > 0;
    if (hasCommand || hasArgs) {
      return await AcpService.discoverModels(providerKey, secretNames, options);
    }
    return await AcpService.discoverModels(providerKey, secretNames);
  } catch {
    // Older agent-servers lack the endpoint.
    return null;
  }
}

/** Where {@link useAcpModelDiscovery}'s ``models`` came from. */
export type AcpModelSource = "live" | "remembered" | "none";

/**
 * The models an ACP provider offers on the active backend, and the one
 * it uses by default, as its own server reports them for the saved
 * credentials. Only local backends can ask before a conversation starts;
 * otherwise ``models`` is the list the agent last reported with the same
 * secrets. ``secretRefs`` limits the lookup to the secrets a launch would
 * receive; null means every saved secret.
 */
export function useAcpModelDiscovery(
  providerKey: string | null | undefined,
  {
    enabled = true,
    secretRefs = null,
    acpCommand = null,
    acpArgs = null,
  }: {
    enabled?: boolean;
    secretRefs?: readonly string[] | null;
    acpCommand?: readonly string[] | null;
    acpArgs?: readonly string[] | null;
  } = {},
) {
  const { backend } = useActiveBackend();
  const isLocal = backend.kind === "local";
  const secrets = useSearchSecrets({ enabled: enabled && isLocal });
  const secretNames = secrets.data
    .map(({ name }) => name)
    .filter((name) => !secretRefs || secretRefs.includes(name))
    .sort();

  const launchKey = acpLaunchKey(providerKey, acpCommand, acpArgs);
  const hasValidCommand =
    providerKey !== "custom" || (!!acpCommand && acpCommand.length > 0);
  const queryEnabled =
    enabled &&
    isLocal &&
    !!providerKey &&
    hasValidCommand &&
    !secrets.isLoading;

  const query = useQuery<ACPModelDiscovery | null, Error>({
    queryKey: [
      ...ACP_MODEL_DISCOVERY_QUERY_KEYS.all,
      backend.id,
      providerKey,
      secretNames,
      acpCommand ?? null,
      acpArgs ?? null,
    ],
    queryFn: () =>
      discover(providerKey as string, secretNames, { acpCommand, acpArgs }),
    enabled: queryEnabled,
    staleTime: 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
    retry: false,
    refetchOnWindowFocus: false,
  });

  const scope = acpModelScope(secretRefs);
  const discovery = queryEnabled ? (query.data ?? null) : null;
  const liveModels = useMemo(
    () => toAcpModelOptions(discovery?.available_models),
    [discovery],
  );
  useEffect(() => {
    if (launchKey) {
      rememberAcpModels(backend.id, launchKey, scope, liveModels);
    }
  }, [backend.id, launchKey, scope, liveModels]);

  const remembered = useRememberedAcpModels(enabled ? launchKey : null, scope);
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
