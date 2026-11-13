import { useMutation } from "@tanstack/react-query";
import { SecretsService } from "#/api/secrets-service";
import type { CustomSecretInput } from "#/api/agent-server-adapter";

export interface UpdateSecretVariables {
  secretToEdit: string;
  name: string;
  description?: string;
  value?: string;
}

/**
 * Update a custom secret's name/description and, when a value is given, its
 * value.
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
export const useUpdateSecret = (options?: {
  onSuccess?: (secret: CustomSecretInput) => void;
  mutationKey?: string[];
}) => {
  const onSuccess = options?.onSuccess;
  return useMutation({
    mutationKey: options?.mutationKey,
    mutationFn: ({
      secretToEdit,
      name,
      description,
      value,
    }: UpdateSecretVariables) =>
      SecretsService.updateSecret(secretToEdit, name, description, value),
    onSuccess: onSuccess
      ? (_data, { name, description }) => onSuccess({ name, description })
      : undefined,
  });
};
