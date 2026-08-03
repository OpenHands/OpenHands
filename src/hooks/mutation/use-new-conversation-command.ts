import { useMutation } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import toast from "react-hot-toast";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToast,
  TOAST_OPTIONS,
} from "#/utils/custom-toast-handlers";
import { useNavigation } from "#/context/navigation-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { getStoredConversationMetadata } from "#/api/conversation-metadata-store";
import { getActiveBackend } from "#/api/backend-registry/active-store";

export const useNewConversationCommand = () => {
  const { navigate } = useNavigation();
  const { t } = useTranslation("openhands");
  const { data: conversation } = useActiveConversation();
  const { backend, orgId } = useActiveBackend();
  const { mutateAsync: createConversation } = useCreateConversation();

  const mutation = useMutation({
    mutationFn: async () => {
      if (!conversation?.id) {
        throw new Error("No active conversation");
      }

      const storedMetadata = getStoredConversationMetadata(conversation.id);
      const workingDir = conversation.workspace?.working_dir?.trim();

      if (backend.kind === "local" && !workingDir) {
        throw new Error("The active conversation workspace is unavailable");
      }

      // Cloud reuses its sandbox id. Local conversations do not expose one,
      // so reuse the exact attached runtime workspace instead of allocating a
      // fresh per-conversation worktree. /new remains independent (no parent).
      return createConversation({
        ...(conversation.selected_repository && conversation.git_provider
          ? {
              repository: {
                name: conversation.selected_repository,
                gitProvider: conversation.git_provider,
                ...(conversation.selected_branch
                  ? { branch: conversation.selected_branch }
                  : {}),
              },
            }
          : {}),
        ...(storedMetadata?.plugins?.length
          ? { plugins: storedMetadata.plugins }
          : {}),
        ...(conversation.launched_agent_profile?.agent_profile_id
          ? {
              agentProfileId:
                conversation.launched_agent_profile.agent_profile_id,
            }
          : {}),
        ...(backend.kind === "cloud"
          ? { sandboxId: conversation.sandbox_id ?? undefined }
          : {
              workingDir,
              // The source path may itself be a worktree, but this operation
              // deliberately attaches that exact existing directory. Passing
              // `new_worktree` would ask the backend to allocate another one
              // and lose uncommitted files from the source runtime.
              workspaceMode: "local_repo",
            }),
        entryPoint: "new_command",
      });
    },
    onMutate: () => {
      const toastId = toast.loading(
        t(I18nKey.CONVERSATION$CLEARING),
        TOAST_OPTIONS,
      );
      return { backendId: backend.id, orgId: orgId ?? null, toastId };
    },
    onSuccess: (data, _variables, invocation) => {
      if (invocation?.toastId) toast.dismiss(invocation.toastId);
      const active = getActiveBackend();
      if (
        invocation?.backendId !== active.backend.id ||
        invocation?.orgId !== (active.orgId ?? null)
      ) {
        return;
      }
      displaySuccessToast(t(I18nKey.CONVERSATION$CLEAR_SUCCESS));
      navigate(`/conversations/${data.conversation_id}`);
    },
    onError: (error, _variables, invocation) => {
      if (invocation?.toastId) toast.dismiss(invocation.toastId);
      const active = getActiveBackend();
      if (
        invocation?.backendId !== active.backend.id ||
        invocation?.orgId !== (active.orgId ?? null)
      ) {
        return;
      }
      let clearError = t(I18nKey.CONVERSATION$CLEAR_UNKNOWN_ERROR);
      if (error instanceof Error) {
        clearError = error.message;
      } else if (typeof error === "string") {
        clearError = error;
      }
      displayErrorToast(
        t(I18nKey.CONVERSATION$CLEAR_FAILED, { error: clearError }),
      );
    },
  });

  return mutation;
};
