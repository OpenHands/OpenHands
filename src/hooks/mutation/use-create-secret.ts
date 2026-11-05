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
 */
export const useCreateSecret = (options?: {
  onSuccess?: (secret: CustomSecretInput) => void;
}) => {
  const onSuccess = options?.onSuccess;
  return useMutation({
    mutationFn: ({ name, value, description }: CreateSecretVariables) =>
      SecretsService.createSecret(name, value, description),
    onSuccess: onSuccess
      ? (_data, { name, description }) => onSuccess({ name, description })
      : undefined,
  });
};
