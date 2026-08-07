import { useMutation, useQueryClient } from "@tanstack/react-query";
import { saveWorkspaceFile } from "#/api/save-workspace-file";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useWorkspaceMutationCounter } from "#/stores/use-workspace-mutation-counter";

export function useSaveWorkspaceFile() {
  const queryClient = useQueryClient();
  const { data: conversation } = useActiveConversation();
  const bump = useWorkspaceMutationCounter((state) => state.bump);

  return useMutation({
    mutationFn: async ({
      relativePath,
      content,
    }: {
      relativePath: string;
      content: string;
    }) => {
      if (!conversation) {
        throw new Error("No active conversation");
      }
      await saveWorkspaceFile({
        conversation,
        relativePath,
        content,
      });
    },
    onSuccess: async () => {
      bump();
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["workspace-file-content"] }),
        queryClient.invalidateQueries({ queryKey: ["workspace-files"] }),
        queryClient.invalidateQueries({ queryKey: ["file_changes"] }),
        queryClient.invalidateQueries({ queryKey: ["file_diff"] }),
      ]);
    },
  });
}
