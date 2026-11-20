import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { FACTORIES_EXPERIMENTAL_FEATURE } from "#/constants/factories";
import { hasAutomationInterface } from "#/manifests/automation-interface";
import { useExperimentalFeatures } from "./use-experimental-features";
import { useDeploymentCapabilities } from "./use-manifest-capabilities";

export const FACTORIES_QUERY_KEY = ["factories"] as const;

/**
 * Whether Factories is on and the active automation service serves factories.
 * Factories is off until the user turns on the `factories` experimental
 * feature on their agent server. `isLoading` is true until the deployment's
 * capabilities and the switch have answered, and a deployment that does not
 * answer counts as unsupported.
 */
export function useFactoriesSupported() {
  const capabilities = useDeploymentCapabilities({
    enabled: hasAutomationInterface(),
  });
  const features = useExperimentalFeatures();
  const isTurnedOn =
    features.data?.some(
      (feature) =>
        feature.key === FACTORIES_EXPERIMENTAL_FEATURE && feature.enabled,
    ) ?? false;
  return {
    isLoading: capabilities.isLoading || features.isLoading,
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
