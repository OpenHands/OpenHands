import { useMutation } from "@tanstack/react-query";
import { SecretsService } from "#/api/secrets-service";

export const useUpdateSecret = () =>
  useMutation({
    mutationFn: ({
      secretToEdit,
      name,
      description,
    }: {
      secretToEdit: string;
      name: string;
      description?: string;
    }) => SecretsService.updateSecret(secretToEdit, name, description),
  });
