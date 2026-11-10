import { useEffect, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import type { ACPModelDiscovery } from "@openhands/typescript-client";
import AcpService from "#/api/acp-service/acp-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useSearchSecrets } from "#/hooks/query/use-get-secrets";
import { ACP_MODEL_DISCOVERY_QUERY_KEYS } from "#/hooks/query/query-keys";
import {
  toAcpEffortChoices,
  toAcpModelOptions,
  type ACPEffortChoice,
  type ACPModelOption,
} from "#/constants/acp-providers";
import { useRememberedAcpModels } from "#/hooks/use-remembered-acp-models";
import {
  acpLaunchArgv,
  isAcpLaunchDiscoverable,
  type AcpLaunch,
} from "#/utils/acp-launch";
import {
  acpModelScope,
  rememberAcpModels,
} from "#/utils/remembered-acp-models";

async function discover(
  launch: AcpLaunch | null,
  secretNames: string[],
): Promise<ACPModelDiscovery | null> {
  if (!launch) return null;
  try {
    return await AcpService.discoverModels(launch, secretNames);
  } catch {
    // Older agent-servers lack the endpoint, and Docker runtime answers 501.
    return null;
  }
}

/** Where {@link useAcpModelDiscovery}'s ``models`` came from. */
export type AcpModelSource = "live" | "remembered" | "none";

/**
 * The models an ACP launch offers on the active backend, and the one it uses by
 * default, as its own server reports them for the saved credentials.
 *
 * `launch` carries the server key **and** the effective command, so a custom
 * profile is asked with the same argv it would spawn. A launch with no command
 * (or one the agent-server cannot resolve) is skipped: no request is made for a
 * custom profile without a command. Only local backends can ask before a
 * conversation starts; otherwise ``models`` is the list the agent last reported
 * with the same secrets and command. ``secretRefs`` limits the lookup to the
 * secrets a launch would receive; null means every saved secret.
 */
export function useAcpModelDiscovery(
  launch: AcpLaunch | null | undefined,
  {
    enabled = true,
    secretRefs = null,
  }: { enabled?: boolean; secretRefs?: readonly string[] | null } = {},
) {
  const { backend } = useActiveBackend();
  const isLocal = backend.kind === "local";
  const activeLaunch = isAcpLaunchDiscoverable(launch) ? launch : null;
  const discoverable = !!activeLaunch;
  const server = activeLaunch?.server ?? null;
  const secrets = useSearchSecrets({
    enabled: enabled && isLocal && discoverable,
  });
  const secretNames = secrets.data
    .map(({ name }) => name)
    .filter((name) => !secretRefs || secretRefs.includes(name))
    .sort();
  // The full argv is the launch's identity: two custom profiles with different
  // commands must not share a cache entry or a remembered list.
  const argv = activeLaunch ? acpLaunchArgv(activeLaunch) : [];
  const queryEnabled = enabled && isLocal && discoverable && !secrets.isLoading;

  const query = useQuery<ACPModelDiscovery | null, Error>({
    queryKey: [
      ...ACP_MODEL_DISCOVERY_QUERY_KEYS.all,
      backend.id,
      activeLaunch,
      secretNames,
    ],
    queryFn: () => discover(activeLaunch, secretNames),
    enabled: queryEnabled,
    staleTime: 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
    retry: false,
    refetchOnWindowFocus: false,
  });

  const scope = acpModelScope(secretRefs, argv);
  const discovery = queryEnabled ? (query.data ?? null) : null;
  const liveModels = useMemo(
    () => toAcpModelOptions(discovery?.available_models),
    [discovery],
  );
  useEffect(() => {
    if (server) {
      rememberAcpModels(backend.id, server, scope, liveModels);
    }
  }, [backend.id, server, scope, liveModels]);

  const remembered = useRememberedAcpModels(enabled ? server : null, scope);
  let models: ACPModelOption[] = [];
  let source: AcpModelSource = "none";
  if (liveModels.length) {
    models = liveModels;
    source = "live";
  } else if (remembered.length) {
    models = remembered;
    source = "remembered";
  }

  const effortsByModel = useMemo(() => {
    const byModel: Record<string, ACPEffortChoice[]> = {};
    for (const [modelId, options] of Object.entries(
      discovery?.efforts_by_model ?? {},
    )) {
      byModel[modelId] = toAcpEffortChoices(options);
    }
    return byModel;
  }, [discovery]);
  const defaultModelId = discovery?.current_model_id ?? null;

  return {
    discovery,
    models,
    source,
    defaultModelId,
    /** The agent's own effort for ``defaultModelId``. */
    defaultEffort: discovery?.current_effort ?? null,
    /** Effort levels ``modelId`` (null: the agent's default model) offers. */
    effortsFor: (modelId: string | null): ACPEffortChoice[] =>
      effortsByModel[modelId ?? defaultModelId ?? ""] ?? [],
    isDiscovering: queryEnabled && query.isFetching && !query.data,
  };
}
