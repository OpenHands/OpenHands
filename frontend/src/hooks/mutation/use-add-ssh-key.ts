import { useMutation, useQueryClient } from "@tanstack/react-query";
import SSHKeysService, { SSHKeyRequest, SSHKey } from "#/api/ssh-keys-service";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useAddSSHKey = () => {
  const queryClient = useQueryClient();
  const { organizationId } = useSelectedOrganizationId();

  return useMutation({
    mutationFn: (request: SSHKeyRequest) => SSHKeysService.addSSHKey(request),
    onSuccess: (newKey: SSHKey) => {
      queryClient.setQueryData<SSHKey[]>(
        ["ssh-keys", organizationId],
        (oldKeys) => [...(oldKeys || []), newKey],
      );
    },
  });
};
