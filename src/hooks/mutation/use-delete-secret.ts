import { useMutation } from "@tanstack/react-query";
import { SecretsService } from "#/api/secrets-service";

export const useDeleteSecret = () =>
  useMutation({
    mutationFn: (id: string) => SecretsService.deleteSecret(id),
  });
