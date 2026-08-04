import { useState } from "react";
import toast from "react-hot-toast";
import { useTranslation } from "react-i18next";
import { CustomChatInput } from "#/components/features/chat/custom-chat-input";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { useLocalWorkspaces } from "#/hooks/query/use-local-workspaces";
import { useModelInterceptor } from "#/hooks/chat/use-model-interceptor";
import { useLlmConfigured } from "#/hooks/use-llm-configured";
import { HOME_PROMPT_DRAFT_KEY } from "#/hooks/chat/use-draft-persistence";
import { useChatAttachmentUpload } from "#/hooks/chat/use-chat-attachment-upload";
import { useConversationStore } from "#/stores/conversation-store";
import type { WorkspaceMode } from "#/api/conversation-metadata-store";
import { setPendingTaskAttachments } from "#/stores/pending-task-attachments-store";
import { enqueueHomeTaskPendingMessage } from "#/utils/enqueue-home-task-pending-message";
import { sendMessageWithAttachments } from "#/utils/send-message-with-attachments";
import { useNavigation } from "#/context/navigation-context";
import { useIsCreatingConversation } from "#/hooks/use-is-creating-conversation";
import { Branch, GitRepository } from "#/types/git";
import { Provider } from "#/types/settings";
import { LocalWorkspace } from "#/types/workspace";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  TOAST_OPTIONS,
} from "#/utils/custom-toast-handlers";
import { getWorkspacesUnsupportedMessage } from "#/utils/workspaces-compatibility";
import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";
import { PluginPickerModal } from "#/components/features/plugins/plugin-picker-modal";
import { PluginPickerTrigger } from "#/components/features/plugins/plugin-picker-trigger";
import { HomeHeaderTitle } from "./home-header/home-header-title";
import { OpenLauncherButton } from "./open-launcher-button";
import { OpenWorkspaceDialog } from "./open-workspace-dialog";
import { OpenRepositoryDialog } from "./open-repository-dialog";
import { HomeGitControlBarPreview } from "./home-git-control-bar-preview";

export function HomeChatLauncher() {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  const { navigate } = useNavigation();
  const isLocal = backend.kind === "local";

  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [pendingWorkspace, setPendingWorkspace] =
    useState<LocalWorkspace | null>(null);
  const [pendingRepository, setPendingRepository] =
    useState<GitRepository | null>(null);
  const [pendingBranch, setPendingBranch] = useState<Branch | null>(null);
  const [pendingProvider, setPendingProvider] = useState<Provider | null>(null);
  const [workspaceMode, setWorkspaceMode] =
    useState<WorkspaceMode>("local_repo");
  const [selectedPlugins, setSelectedPlugins] = useState<PluginSpec[]>([]);
  const [isPluginPickerOpen, setIsPluginPickerOpen] = useState(false);

  const { mutateAsync: createConversation, isPending } =
    useCreateConversation();
  const isCreatingElsewhere = useIsCreatingConversation();
  const isCreating = isPending || isCreatingElsewhere;
  const { isConfigured: isLlmConfigured, isLoading: isLlmConfigLoading } =
    useLlmConfigured();
  // Block sending entirely when there's no usable LLM; the banner above the
  // launcher (rendered by the home route) explains it and offers setup.
  const llmBlocked = !isLlmConfigLoading && !isLlmConfigured;
  const { images, files, imagesMarkedUploadAsFile, clearAllFiles } =
    useConversationStore();
  const { handleUpload } = useChatAttachmentUpload();
  const { error: workspacesError } = useLocalWorkspaces({ enabled: isLocal });
  const workspacesUnsupportedMessage = isLocal
    ? getWorkspacesUnsupportedMessage(workspacesError, t)
    : null;

  const hasSelection = isLocal
    ? !!pendingWorkspace
    : !!pendingRepository && !!pendingBranch;

  const handleSubmit = (message: string) => {
    const trimmed = message.trim();
    const hasAttachments = images.length > 0 || files.length > 0;
    if ((!trimmed && !hasAttachments) || isCreating) return;

    // Safety net: the input is disabled when there's no usable LLM, but never
    // create a conversation that can't run (it would fail with a cryptic
    // API-key error on the first turn).
    if (llmBlocked) return;

    const attachmentSnapshot = {
      images: [...images],
      files: [...files],
    };

    // Workspace/repo are optional — match the "Start from scratch" flow which
    // creates a conversation with no working dir and no repo. Build the
    // payload from whatever is selected.
    // When attachments are present the first user message is sent afterward
    // via sendMessageWithAttachments / flushPendingTaskAttachments. Passing
    // query here would create a duplicate text-only initial_message.
    let variables: Parameters<typeof createConversation>[0] = {
      query: hasAttachments ? undefined : trimmed || undefined,
      entryPoint: "home_chat_launcher",
    };
    if (isLocal && pendingWorkspace) {
      variables = {
        ...variables,
        workingDir: pendingWorkspace.path,
        workspaceMode,
      };
    } else if (!isLocal && pendingRepository && pendingBranch) {
      variables = {
        ...variables,
        repository: {
          name: pendingRepository.full_name,
          gitProvider: pendingRepository.git_provider,
          branch: pendingBranch.name,
        },
      };
    }

    // Explicitly-attached plugins are additive on top of any ambient set and
    // are resolved from git at run time. Omitted entirely when none selected so
    // nothing attaches unless the user picked it.
    if (selectedPlugins.length > 0) {
      variables = { ...variables, plugins: selectedPlugins };
    }

    // Loading toast gives the user a clear signal that the request is in
    // flight; dismissed precisely once the mutation resolves.
    const toastId = toast.loading(
      t(I18nKey.HOME$CREATING_CONVERSATION),
      TOAST_OPTIONS,
    );

    void (async () => {
      try {
        const data = await createConversation(variables);
        toast.dismiss(toastId);
        try {
          sessionStorage.removeItem(HOME_PROMPT_DRAFT_KEY);
        } catch {
          // sessionStorage not available
        }
        const targetConversationId = data.conversation_id;
        const isTaskConversation = targetConversationId.startsWith("task-");

        if (hasAttachments) {
          // Cloud sandboxes provision asynchronously; uploads and the first
          // message must target the runtime URL, not the bundled local server.
          const shouldDeferAttachments = !isLocal || isTaskConversation;

          if (shouldDeferAttachments) {
            const taskId =
              data.task_id ??
              (isTaskConversation
                ? targetConversationId.slice("task-".length)
                : null);

            if (!taskId) {
              displayErrorToast(null);
              return;
            }

            setPendingTaskAttachments(taskId, {
              content: trimmed,
              images: attachmentSnapshot.images,
              files: attachmentSnapshot.files,
              imagesMarkedUploadAsFile: [...imagesMarkedUploadAsFile],
            });
            clearAllFiles();
            await enqueueHomeTaskPendingMessage({
              conversationId: targetConversationId,
              text: trimmed,
              images: attachmentSnapshot.images,
              imagesMarkedUploadAsFile,
            });
            navigate(`/conversations/${targetConversationId}`);
            return;
          } else {
            try {
              await sendMessageWithAttachments({
                conversationId: targetConversationId,
                content: trimmed,
                images: attachmentSnapshot.images,
                files: attachmentSnapshot.files,
                imagesMarkedUploadAsFile,
                t,
              });
              clearAllFiles();
            } catch (error) {
              displayErrorToast(error instanceof Error ? error.message : null);
              return;
            }
          }
        }

        if (isTaskConversation && trimmed) {
          await enqueueHomeTaskPendingMessage({
            conversationId: targetConversationId,
            text: trimmed,
            images: [],
            imagesMarkedUploadAsFile: [],
          });
        }

        navigate(`/conversations/${targetConversationId}`);
      } catch (error) {
        toast.dismiss(toastId);
        displayErrorToast(error instanceof Error ? error.message : null);
      }
    })();
  };

  // Without this wrapper a `/model NAME` typed here would become the first
  // user message of the new conversation. The interceptor activates the
  // profile globally (null conversationId path) so the next conversation
  // launches with it.
  const handleSubmitWithModelGuard = useModelInterceptor(null, handleSubmit);

  return (
    <div
      data-testid="home-chat-launcher"
      className="flex w-full max-w-[800px] flex-col gap-4 md:px-4"
    >
      <div className="flex w-full justify-center">
        <HomeHeaderTitle />
      </div>

      <div className="w-full">
        <CustomChatInput
          onSubmit={handleSubmitWithModelGuard}
          onFilesPaste={handleUpload}
          disabled={isCreating || llmBlocked}
        />
      </div>

      <div className="flex items-center justify-start gap-2">
        {hasSelection ? (
          <HomeGitControlBarPreview
            workspace={pendingWorkspace}
            repository={pendingRepository}
            branch={pendingBranch}
            provider={pendingProvider}
            workspaceMode={workspaceMode}
            backendKind={backend.kind}
            onRepoClick={() => setIsDialogOpen(true)}
            onWorkspaceModeChange={setWorkspaceMode}
          />
        ) : (
          <OpenLauncherButton
            kind={isLocal ? "local" : "cloud"}
            onClick={() => setIsDialogOpen(true)}
            disabled={isCreating || Boolean(workspacesUnsupportedMessage)}
            disabledTooltip={workspacesUnsupportedMessage}
          />
        )}
        <PluginPickerTrigger
          count={selectedPlugins.length}
          onClick={() => setIsPluginPickerOpen(true)}
          disabled={isCreating}
        />
      </div>

      {isLocal ? (
        <OpenWorkspaceDialog
          isOpen={isDialogOpen}
          onClose={() => setIsDialogOpen(false)}
          onConfirm={(workspace) => {
            setPendingWorkspace(workspace);
            setPendingRepository(null);
            setPendingBranch(null);
            setPendingProvider(null);
            setWorkspaceMode("local_repo");
          }}
        />
      ) : (
        <OpenRepositoryDialog
          isOpen={isDialogOpen}
          onClose={() => setIsDialogOpen(false)}
          onConfirm={({ repository, branch, provider }) => {
            setPendingRepository(repository);
            setPendingBranch(branch);
            setPendingProvider(provider ?? repository.git_provider);
            setPendingWorkspace(null);
            setWorkspaceMode("local_repo");
          }}
        />
      )}

      {isPluginPickerOpen && (
        <PluginPickerModal
          selected={selectedPlugins}
          onChange={setSelectedPlugins}
          onClose={() => setIsPluginPickerOpen(false)}
        />
      )}
    </div>
  );
}
