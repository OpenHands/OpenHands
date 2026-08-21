import { useMutation } from "@tanstack/react-query";
import { SecretsService } from "#/api/secrets-service";
import { removeSecretFromActiveConversation } from "#/api/conversation-service/conversation-secret-sync";

export const useDeleteSecret = () =>
  useMutation({
    mutationFn: async (id: string) => {
      const res = await SecretsService.deleteSecret(id);
      await removeSecretFromActiveConversation(id);
      return res;
    },
  });
