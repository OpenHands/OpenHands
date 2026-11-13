import { useMutation } from "@tanstack/react-query";
import { SecretsService } from "#/api/secrets-service";
import type { CustomSecretInput } from "#/api/agent-server-adapter";

export interface CreateSecretVariables {
  name: string;
  value: string;
  description?: string;
}

/**
 * Create (or upsert) a custom secret.
 *
 * `options.onSuccess` runs on the mutation itself, not as a per-call
 * `mutate(..., { onSuccess })`. TanStack Query drops per-call callbacks with
 * the unmounted observer, so a caller that can unmount while the save is in
 * flight — the chat's virtualized list remounting a secret-request card across
 * the virtualization threshold — would otherwise never see the save settle.
 *
 * `options.mutationKey` lets a caller that can remount read the in-flight state
 * with `useIsMutating` (see `secretRequestMutationKey`), which lives on the
 * cache entry rather than a per-observer `isPending`.
 */
export const useCreateSecret = (options?: {
  onSuccess?: (secret: CustomSecretInput) => void;
  mutationKey?: string[];
}) => {
  const onSuccess = options?.onSuccess;
  return useMutation({
    mutationKey: options?.mutationKey,
    mutationFn: ({ name, value, description }: CreateSecretVariables) =>
      SecretsService.createSecret(name, value, description),
    onSuccess: onSuccess
      ? (_data, { name, description }) => onSuccess({ name, description })
      : undefined,
  });
};
