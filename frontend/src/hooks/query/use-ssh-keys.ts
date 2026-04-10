import { useQuery } from "@tanstack/react-query";
import SSHKeysService from "#/api/ssh-keys-service";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useSSHKeys = () => {
  const { organizationId } = useSelectedOrganizationId();

  return useQuery({
    queryKey: ["ssh-keys", organizationId],
    queryFn: () => SSHKeysService.listSSHKeys(),
  });
};
