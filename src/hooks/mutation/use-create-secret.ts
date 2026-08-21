import { useMutation } from "@tanstack/react-query";
import { SecretsService } from "#/api/secrets-service";
import { syncSecretToActiveConversation } from "#/api/conversation-service/conversation-secret-sync";

export const useCreateSecret = () =>
  useMutation({
    mutationFn: async ({
      name,
      value,
      description,
    }: {
      name: string;
      value: string;
      description?: string;
    }) => {
      const res = await SecretsService.createSecret(name, value, description);
      await syncSecretToActiveConversation(name, description);
      return res;
    },
  });
