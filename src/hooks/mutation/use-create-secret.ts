import { useMutation } from "@tanstack/react-query";
import { SecretsService } from "#/api/secrets-service";

export const useCreateSecret = () =>
  useMutation({
    mutationFn: ({
      name,
      value,
      description,
    }: {
      name: string;
      value: string;
      description?: string;
    }) => SecretsService.createSecret(name, value, description),
  });
