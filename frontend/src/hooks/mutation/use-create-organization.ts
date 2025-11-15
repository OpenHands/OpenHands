import { useMutation, useQueryClient } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";

export const useCreateOrganization = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ name }: { name: string }) =>
      organizationService.createOrganization({ name }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["organizations"] });
    },
  });
};
