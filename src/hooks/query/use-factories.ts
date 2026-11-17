import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { hasAutomationInterface } from "#/manifests/automation-interface";
import { useDeploymentCapabilities } from "./use-manifest-capabilities";

export const FACTORIES_QUERY_KEY = ["factories"] as const;

/**
 * Whether the active automation service serves factories. `isLoading` is true
 * until the deployment's capabilities have answered, and a deployment that
 * does not answer counts as unsupported.
 */
export function useFactoriesSupported() {
  const { data, isLoading } = useDeploymentCapabilities({
    enabled: hasAutomationInterface(),
  });
  return {
    isLoading,
    isSupported: AutomationService.supportsFactories(data),
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
