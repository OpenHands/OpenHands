import { useMutation } from "@tanstack/react-query";
import { SecretsService } from "#/api/secrets-service";
import {
  renameSecretInActiveConversation,
  syncSecretToActiveConversation,
} from "#/api/conversation-service/conversation-secret-sync";

export const useUpdateSecret = () =>
  useMutation({
    mutationFn: async ({
      secretToEdit,
      name,
      description,
      value,
    }: {
      secretToEdit: string;
      name: string;
      description?: string;
      value?: string;
    }) => {
      const res = await SecretsService.updateSecret(
        secretToEdit,
        name,
        description,
        value,
      );
      if (secretToEdit !== name) {
        await renameSecretInActiveConversation(secretToEdit, name, description);
      } else {
        await syncSecretToActiveConversation(name, description);
      }
      return res;
    },
  });
