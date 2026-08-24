import { useEffect, useState } from "react";
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
import { RecommendedAutomationsLauncher } from "#/components/features/automations/recommended-automations-launcher";
import { PinnedAutomationsDashboard } from "./featured-automations/pinned-automations-dashboard";
import { RunningAutomationsList } from "./featured-automations/running-automations-list";
import { HomeHeaderTitle } from "./home-header/home-header-title";
import { OpenLauncherButton } from "./open-launcher-button";
import { OpenWorkspaceDialog } from "./open-workspace-dialog";
import { OpenRepositoryDialog } from "./open-repository-dialog";
import { HomeGitControlBarPreview } from "./home-git-control-bar-preview";

/**
 * Fixed id for the "Creating conversation…" toast.
 *
 * react-hot-toast gives loading toasts an infinite duration, so this one only
 * ever leaves the screen when something dismisses it by id. Under a generated
 * id a toast that outlives its dismiss becomes unreachable — no other code can
 * name it — and stays up until the page is reloaded (#15701). A fixed id keeps
 * it addressable, and means a repeat submit reuses the same toast instead of
 * stacking another one that also has to be cleaned up.
 */
const CREATING_CONVERSATION_TOAST_ID = "creating-conversation";

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

  // The submit below drives a promise chain that isn't tied to this component,
  // and the create mutation stays pending past the point where the conversation
  // itself exists (it also resolves agent profiles and snapshots installed
  // plugins). Leaving the home page inside that window — clicking a sidebar
  // conversation, switching backend — tears the launcher down while the chain
  // is still awaiting, so clear the toast on the way out rather than trusting
  // the chain to get there.
  useEffect(() => () => toast.dismiss(CREATING_CONVERSATION_TOAST_ID), []);

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
    // flight; dismissed as soon as the conversation exists, and again in the
    // `finally` below so no exit path can leave it behind.
    toast.loading(t(I18nKey.HOME$CREATING_CONVERSATION), {
      ...TOAST_OPTIONS,
      id: CREATING_CONVERSATION_TOAST_ID,
    });

    void (async () => {
      try {
        const data = await createConversation(variables);
        // The conversation exists from here on, so drop the toast before the
        // attachment / pending-message work instead of leaving a "Creating
        // conversation…" label up for a step that isn't creating anything.
        toast.dismiss(CREATING_CONVERSATION_TOAST_ID);
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
        displayErrorToast(error instanceof Error ? error.message : null);
      } finally {
        // Safety net for every path above, including the early returns. The
        // toast has no duration of its own, so a single missed dismiss leaves
        // it on screen for the rest of the session.
        toast.dismiss(CREATING_CONVERSATION_TOAST_ID);
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
      className="flex w-full flex-col items-center pt-[max(4rem,28vh)] pb-10"
    >
      <div className="flex w-full max-w-[800px] flex-col gap-4 md:px-4">
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

        <div className="mt-8 flex w-full flex-col gap-8">
          <RecommendedAutomationsLauncher variant="rail" />
          <PinnedAutomationsDashboard />
          <RunningAutomationsList />
        </div>
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
