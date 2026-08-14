import { useMutation, useQueryClient } from "@tanstack/react-query";
import ModelProvidersService from "#/api/model-providers-service";
import type {
  CreateProviderRequest,
  ModelPayload,
  TestProviderResponse,
  UpdateProviderRequest,
} from "#/api/model-providers-service";
import { MODEL_PROVIDERS_QUERY_KEYS } from "#/hooks/query/query-keys";

/** Shared invalidation: every provider mutation refreshes the provider list. */
function useInvalidateProviders() {
  const queryClient = useQueryClient();
  return () =>
    queryClient.invalidateQueries({
      queryKey: MODEL_PROVIDERS_QUERY_KEYS.all,
    });
}

export function useCreateModelProvider() {
  const invalidate = useInvalidateProviders();
  return useMutation({
    mutationFn: (request: CreateProviderRequest) =>
      ModelProvidersService.createProvider(request),
    onSuccess: invalidate,
    meta: { disableToast: true },
  });
}

export function useUpdateModelProvider() {
  const invalidate = useInvalidateProviders();
  return useMutation({
    mutationFn: ({
      id,
      request,
    }: {
      id: string;
      request: UpdateProviderRequest;
    }) => ModelProvidersService.updateProvider(id, request),
    onSuccess: invalidate,
    meta: { disableToast: true },
  });
}

export function useDeleteModelProvider() {
  const invalidate = useInvalidateProviders();
  return useMutation({
    mutationFn: (id: string) => ModelProvidersService.deleteProvider(id),
    onSuccess: invalidate,
    meta: { disableToast: true },
  });
}

export function useAddProviderModel() {
  const invalidate = useInvalidateProviders();
  return useMutation({
    mutationFn: ({ id, model }: { id: string; model: ModelPayload }) =>
      ModelProvidersService.addModel(id, model),
    onSuccess: invalidate,
    meta: { disableToast: true },
  });
}

export function useUpdateProviderModel() {
  const invalidate = useInvalidateProviders();
  return useMutation({
    mutationFn: ({
      id,
      modelName,
      model,
    }: {
      id: string;
      modelName: string;
      model: ModelPayload;
    }) => ModelProvidersService.updateModel(id, modelName, model),
    onSuccess: invalidate,
    meta: { disableToast: true },
  });
}

export function useRemoveProviderModel() {
  const invalidate = useInvalidateProviders();
  return useMutation({
    mutationFn: ({ id, modelName }: { id: string; modelName: string }) =>
      ModelProvidersService.removeModel(id, modelName),
    onSuccess: invalidate,
    meta: { disableToast: true },
  });
}

export function useTestModelProvider() {
  const invalidate = useInvalidateProviders();
  return useMutation({
    mutationFn: (id: string) => ModelProvidersService.testProvider(id),
    onSuccess: invalidate,
    meta: { disableToast: true },
  });
}

export type { TestProviderResponse };
