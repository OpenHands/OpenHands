import { useMutation, useQueryClient } from "@tanstack/react-query";
import SSHKeysService, { SSHKey } from "#/api/ssh-keys-service";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useDeleteSSHKey = () => {
  const queryClient = useQueryClient();
  const { organizationId } = useSelectedOrganizationId();

  return useMutation({
    mutationFn: (keyId: string) => SSHKeysService.deleteSSHKey(keyId),
    onSuccess: (_data, keyId) => {
      queryClient.setQueryData<SSHKey[]>(
        ["ssh-keys", organizationId],
        (oldKeys) => (oldKeys || []).filter((key) => key.id !== keyId),
      );
    },
  });
};
