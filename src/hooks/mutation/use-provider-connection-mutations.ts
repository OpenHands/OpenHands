import { useMutation, useQueryClient } from "@tanstack/react-query";
import ProviderConnectionsService from "#/api/provider-connections-service";
import type {
  CreateConnectionRequest,
  CreateProfileFromConnectionRequest,
  ProfileFromConnectionResponse,
  UpdateConnectionRequest,
  ValidateConnectionResponse,
} from "#/api/provider-connections-service";
import { PROVIDER_CONNECTIONS_QUERY_KEYS } from "#/hooks/query/query-keys";

export function useCreateProviderConnection() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: CreateConnectionRequest) =>
      ProviderConnectionsService.createConnection(request),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: PROVIDER_CONNECTIONS_QUERY_KEYS.all,
      });
    },
    meta: { disableToast: true },
  });
}

export function useUpdateProviderConnection() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      request,
    }: {
      id: string;
      request: UpdateConnectionRequest;
    }) => ProviderConnectionsService.updateConnection(id, request),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: PROVIDER_CONNECTIONS_QUERY_KEYS.all,
      });
    },
    meta: { disableToast: true },
  });
}

export function useDeleteProviderConnection() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => ProviderConnectionsService.deleteConnection(id),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: PROVIDER_CONNECTIONS_QUERY_KEYS.all,
      });
    },
    meta: { disableToast: true },
  });
}

export function useValidateProviderConnection() {
  const queryClient = useQueryClient();
  return useMutation({
    // Both the wizard "Test connection" and the row "Refresh" are explicit,
    // user-triggered checks, so probe the key live to get an honest `verified`
    // flag rather than a catalog-only lookup.
    mutationFn: (id: string) =>
      ProviderConnectionsService.validateConnection(id, { live: true }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: PROVIDER_CONNECTIONS_QUERY_KEYS.all,
      });
    },
    meta: { disableToast: true },
  });
}

export function useCreateProfileFromConnection() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      request,
    }: {
      id: string;
      request: CreateProfileFromConnectionRequest;
    }) => ProviderConnectionsService.createProfileFromConnection(id, request),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: PROVIDER_CONNECTIONS_QUERY_KEYS.all,
      });
    },
    meta: { disableToast: true },
  });
}

export type { ValidateConnectionResponse, ProfileFromConnectionResponse };
