import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { FACTORIES_EXPERIMENTAL_FEATURE } from "#/constants/factories";
import { hasAutomationInterface } from "#/manifests/automation-interface";
import { useCloudIntegrationsConfig } from "./use-cloud-integrations-config";
import { useExperimentalFeatures } from "./use-experimental-features";
import { useDeploymentCapabilities } from "./use-manifest-capabilities";

export const FACTORIES_QUERY_KEY = ["factories"] as const;

/**
 * Whether Factories is on and the active automation service serves factories.
 * Each deployment has one switch: on a cloud backend it is the instance's
 * `enable_factories` flag, and on a local backend it is the agent server's
 * `factories` experimental feature. The other switch is never read.
 * `isLoading` is true until the deployment's capabilities and the switch have
 * answered, and a deployment that does not answer counts as unsupported.
 */
export function useFactoriesSupported() {
  const { backend } = useActiveBackend();
  const isCloud = backend.kind === "cloud";
  const capabilities = useDeploymentCapabilities({
    enabled: hasAutomationInterface(),
  });
  const cloudConfig = useCloudIntegrationsConfig();
  const features = useExperimentalFeatures();
  const isTurnedOn = isCloud
    ? cloudConfig.data?.isFactoriesEnabled === true
    : (features.data?.some(
        (feature) =>
          feature.key === FACTORIES_EXPERIMENTAL_FEATURE && feature.enabled,
      ) ?? false);
  const isSwitchLoading = isCloud ? cloudConfig.isLoading : features.isLoading;
  return {
    isLoading: capabilities.isLoading || isSwitchLoading,
    isSupported:
      isTurnedOn && AutomationService.supportsFactories(capabilities.data),
  };
}

/** The org's factories, newest first. */
export function useFactories({ enabled = true } = {}) {
  const active = useActiveBackend();
  return useQuery({
    queryKey: [...FACTORIES_QUERY_KEY, active.backend.id, active.orgId],
    queryFn: () => AutomationService.listFactories(),
    select: (response) => response.factories,
    staleTime: 0,
    enabled,
  });
}

export function useFactory(id: string, { enabled = true } = {}) {
  const active = useActiveBackend();
  return useQuery({
    queryKey: [...FACTORIES_QUERY_KEY, id, active.backend.id, active.orgId],
    queryFn: () => AutomationService.getFactory(id),
    staleTime: 0,
    enabled: enabled && id !== "",
  });
}

export function useCreateFactory() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (name: string) => AutomationService.createFactory(name),
    meta: { disableToast: true },
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: FACTORIES_QUERY_KEY }),
  });
}

export function useRenameFactory() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) =>
      AutomationService.renameFactory(id, name),
    meta: { disableToast: true },
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: FACTORIES_QUERY_KEY }),
  });
}

export function useDeleteFactory() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => AutomationService.deleteFactory(id),
    meta: { disableToast: true },
    onSuccess: (_data, id) => {
      // Drop the deleted factory's own query first, so the invalidation below
      // does not refetch it into a 404 while its page is still mounted.
      queryClient.removeQueries({ queryKey: [...FACTORIES_QUERY_KEY, id] });
      void queryClient.invalidateQueries({ queryKey: FACTORIES_QUERY_KEY });
    },
  });
}
