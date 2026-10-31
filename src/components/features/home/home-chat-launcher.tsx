import { useConversationLaunchStore } from "#/stores/conversation-launch-store";
import { useConversationWorkspace } from "#/hooks/query/use-conversation-workspace";
import { useEffect, useRef, useState } from "react";
import toast from "react-hot-toast";
import { useTranslation } from "react-i18next";
import { ConversationStartupProgress } from "#/components/features/chat/conversation-startup-progress";
import { CustomChatInput } from "#/components/features/chat/custom-chat-input";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { useLocalWorkspaces } from "#/hooks/query/use-local-workspaces";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { useModelInterceptor } from "#/hooks/chat/use-model-interceptor";
import { useLlmConfigured } from "#/hooks/use-llm-configured";
import { HOME_PROMPT_DRAFT_KEY } from "#/hooks/chat/use-draft-persistence";
import { useChatAttachmentUpload } from "#/hooks/chat/use-chat-attachment-upload";
import { useConversationStore } from "#/stores/conversation-store";
import { useModelStore } from "#/stores/model-store";
import type { WorkspaceMode } from "#/api/conversation-metadata-store";
import { setPendingTaskAttachments } from "#/stores/pending-task-attachments-store";
import { enqueueHomeTaskPendingMessage } from "#/utils/enqueue-home-task-pending-message";
import { sendMessageWithAttachments } from "#/utils/send-message-with-attachments";
import { useNavigation } from "#/context/navigation-context";
import { useIsCreatingConversation } from "#/hooks/use-is-creating-conversation";
import { useRememberedRepositorySelection } from "#/hooks/use-remembered-repository-selection";
import { LocalWorkspace } from "#/types/workspace";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import {
  displayErrorToast,
  TOAST_OPTIONS,
} from "#/utils/custom-toast-handlers";
import { getApiErrorMessage } from "#/utils/api-error-message";
import { getWorkspacesUnsupportedMessage } from "#/utils/workspaces-compatibility";
import {
  DEFAULT_WORKSPACE_MODE,
  getDefaultWorkspaceMode,
  getWorkspaceModeOptions,
  readStoredLocalWorkspaceMode,
  writeStoredLocalWorkspaceMode,
} from "#/utils/workspace-mode";
import { useSettings } from "#/hooks/query/use-settings";
import { WorkspaceModeSelector } from "#/components/features/chat/workspace-mode-selector";
import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";
import { markAutomationSetupHandoff } from "#/api/automation-setup-handoff-store";
import { PluginPickerModal } from "#/components/features/plugins/plugin-picker-modal";
import { PluginPickerTrigger } from "#/components/features/plugins/plugin-picker-trigger";
import { RecommendedAutomationsLauncher } from "#/components/features/automations/recommended-automations-launcher";
import { useHomeRecommendedAutomationsDismissed } from "./use-home-recommended-automations-dismissed";
import { writeHomeRecommendedAutomationsDismissed } from "./home-recommended-automations-storage";
import { PinnedAutomationsDashboard } from "./featured-automations/pinned-automations-dashboard";
import { RunningAutomationsList } from "./featured-automations/running-automations-list";
import { HomeHeaderTitle } from "./home-header/home-header-title";
import {
  HomeLauncherModeToggle,
  type HomeLauncherMode,
} from "./home-launcher-mode-toggle";
import { OpenLauncherButton } from "./open-launcher-button";
import { OpenWorkspaceDialog } from "./open-workspace-dialog";
import { OpenRepositoryDialog } from "./open-repository-dialog";
import { HomeGitControlBarPreview } from "./home-git-control-bar-preview";

// How long a create may run before the "Creating conversation…" toast appears.
const CREATING_TOAST_DELAY_MS = 600;
// Once shown, how long it stays up at least, so it never flashes.
const CREATING_TOAST_MIN_VISIBLE_MS = 700;

export function HomeChatLauncher() {
  const { t } = useTranslation("openhands");
  const { backend, orgId } = useActiveBackend();
  const { navigate, currentPath } = useNavigation();
  const ownerKey = JSON.stringify([backend.id, orgId, currentPath]);
  const currentOwner = useRef(ownerKey);
  currentOwner.current = ownerKey;
  const launchInFlight = useRef(false);
  const [pendingLaunchId, setPendingLaunchId] = useState<string | null>(null);
  const launch = useConversationLaunchStore((state) =>
    pendingLaunchId ? state.launches[pendingLaunchId] : null,
  );
  const activeLaunch =
    launch?.backendId === backend.id && launch.orgId === (orgId ?? null)
      ? launch
      : null;
  const isLocal = backend.kind === "local";

  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [pendingWorkspace, setPendingWorkspace] =
    useState<LocalWorkspace | null>(null);
  const {
    selection: repositorySelection,
    setSelection: setRepositorySelection,
    clearSelection: clearRepositorySelection,
  } = useRememberedRepositorySelection(backend.id, !isLocal);
  const { data: settings } = useSettings();
  const [lastUsedWorkspaceMode] = useState(readStoredLocalWorkspaceMode);
  const [pickedWorkspaceMode, setPickedWorkspaceMode] =
    useState<WorkspaceMode | null>(null);
  // Picked here > Settings → Runtime default > last used in this browser.
  const workspaceMode =
    pickedWorkspaceMode ??
    getDefaultWorkspaceMode(
      isLocal ? settings?.conversation_settings?.default_runtime : null,
      lastUsedWorkspaceMode,
    );
  const isRecommendedAutomationsDismissed =
    useHomeRecommendedAutomationsDismissed();
  const [selectedPlugins, setSelectedPlugins] = useState<PluginSpec[]>([]);
  const [isPluginPickerOpen, setIsPluginPickerOpen] = useState(false);
  const [launcherMode, setLauncherMode] = useState<HomeLauncherMode>("code");
  const displayedLauncherMode = launcherMode;
  const isAutomateMode = launcherMode === "automate";
  const handleLauncherModeChange = setLauncherMode;
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  // The launcher reports a failed create itself (see handleSubmit), so the
  // global mutation error toast is turned off to keep it to one toast.
  const { mutateAsync: createConversation, isPending } = useCreateConversation({
    disableToast: true,
  });
  const isCreatingElsewhere = useIsCreatingConversation();
  const isCreating = isPending || isCreatingElsewhere;
  const { isConfigured: isLlmConfigured, isLoading: isLlmConfigLoading } =
    useLlmConfigured();
  // Block sending entirely when there's no usable LLM; the banner above the
  // launcher (rendered by the home route) explains it and offers setup.
  const llmBlocked = !isLlmConfigLoading && !isLlmConfigured;
  const {
    images,
    files,
    imagesMarkedUploadAsFile,
    clearAllFiles,
    restoreMessageToInputIfEmpty,
  } = useConversationStore();
  const { handleUpload } = useChatAttachmentUpload();
  const homeLlmProfile = useModelStore((s) => s.homeLlmProfile);
  const { data: llmProfiles } = useLlmProfiles();
  const { error: workspacesError } = useLocalWorkspaces({ enabled: isLocal });
  const {
    isolated,
    dockerSelectable,
    serverInfoLoading,
    unsupportedMessage: runtimeWorkspaceMessage,
  } = useConversationWorkspace();
  const workspaceModes = getWorkspaceModeOptions(
    dockerSelectable,
    !isLocal || !!pendingWorkspace,
  );
  // A remembered mode the server no longer offers falls back, but only once
  // server_info has answered: a pending probe must not silently turn a Docker
  // choice into Local (a server that cannot run it rejects the start).
  const effectiveWorkspaceMode =
    serverInfoLoading || workspaceModes.includes(workspaceMode)
      ? workspaceMode
      : DEFAULT_WORKSPACE_MODE;
  const workspacesUnsupportedMessage =
    runtimeWorkspaceMessage ??
    (isLocal ? getWorkspacesUnsupportedMessage(workspacesError, t) : null);

  const setWorkspaceMode = (mode: WorkspaceMode) => {
    setPickedWorkspaceMode(mode);
    if (isLocal) writeStoredLocalWorkspaceMode(mode);
  };

  const hasSelection = isLocal ? !!pendingWorkspace : !!repositorySelection;

  const handleSubmit = (message: string) => {
    const trimmed = message.trim();
    const hasAttachments = images.length > 0 || files.length > 0;
    if ((!trimmed && !hasAttachments) || isCreating || launchInFlight.current)
      return;

    // Safety net: the input is disabled when there's no usable LLM, but never
    // create a conversation that can't run (it would fail with a cryptic
    // API-key error on the first turn).
    if (llmBlocked) return;

    launchInFlight.current = true;
    const ownsLaunch = () =>
      isMountedRef.current && currentOwner.current === ownerKey;
    const clientLaunchId = !isLocal
      ? useConversationLaunchStore.getState().begin({
          backendId: backend.id,
          orgId: orgId ?? null,
          message,
          files: [...images, ...files],
          attachments: {
            content: message,
            images: [...images],
            files: [...files],
            imagesMarkedUploadAsFile: [...imagesMarkedUploadAsFile],
          },
        })
      : null;
    if (clientLaunchId) {
      setPendingLaunchId(clientLaunchId);
      clearAllFiles();
    }

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
      automationSetup: isAutomateMode,
      entryPoint: "home_chat_launcher",
    };
    // An isolated backend owns its workspace, so a host selection left over
    // from a non-isolated session must not be forwarded: the server rejects it
    // (`HOME$ISOLATED_WORKSPACE_NOTICE`) and the user sees an error toast for a
    // selection they may not have noticed. Creation proceeds isolated instead;
    // the launcher still offers an explicit "clear" affordance for the UI.
    if (isLocal && pendingWorkspace && !isolated) {
      variables = {
        ...variables,
        workingDir: pendingWorkspace.path,
        workspaceMode: effectiveWorkspaceMode,
      };
    } else if (isLocal && effectiveWorkspaceMode === "docker_container") {
      // No workspace: the container gets its own, so only the runtime is sent.
      variables = { ...variables, workspaceMode: effectiveWorkspaceMode };
    } else if (!isLocal && repositorySelection) {
      variables = {
        ...variables,
        repository: {
          name: repositorySelection.repository.full_name,
          gitProvider: repositorySelection.repository.git_provider,
          branch: repositorySelection.branch.name,
        },
      };
    }

    // Explicitly-attached plugins are additive on top of any ambient set and
    // are resolved from git at run time. Omitted entirely when none selected so
    // nothing attaches unless the user picked it.
    if (!isAutomateMode && selectedPlugins.length > 0) {
      variables = { ...variables, plugins: selectedPlugins };
    }

    // A cloud member's home LLM pick (see useChatInputLlmProfileState), unless
    // that profile has since gone.
    if (
      homeLlmProfile &&
      llmProfiles?.profiles.some((p) => p.name === homeLlmProfile)
    ) {
      variables = { ...variables, llmProfile: homeLlmProfile };
    }

    const openAutomationSetupMode = (conversationId: string) => {
      if (!isAutomateMode) return;
      markAutomationSetupHandoff(conversationId);
    };

    // Most creates resolve quickly and navigate away, so an immediate loading
    // toast only flashes. Show it only once the request is noticeably slow,
    // and once shown keep it up long enough to read. The Toaster is global, so
    // it outlives the navigation without delaying it. A failure removes it at
    // once because the error toast takes its place.
    let toastId: string | undefined;
    let toastShownAt = 0;
    const toastTimer = isLocal
      ? setTimeout(() => {
          toastShownAt = Date.now();
          toastId = toast.loading(
            t(I18nKey.HOME$CREATING_CONVERSATION),
            TOAST_OPTIONS,
          );
        }, CREATING_TOAST_DELAY_MS)
      : undefined;
    const dismissCreatingToast = ({ immediately = false } = {}) => {
      clearTimeout(toastTimer);
      if (!toastId) return;
      const id = toastId;
      const remaining =
        CREATING_TOAST_MIN_VISIBLE_MS - (Date.now() - toastShownAt);
      if (immediately || remaining <= 0) toast.dismiss(id);
      else setTimeout(() => toast.dismiss(id), remaining);
    };

    if (clientLaunchId) navigate(`/conversations/${clientLaunchId}`);

    void (async () => {
      let isCreated = false;
      let createdTarget: string | null = null;
      try {
        const data = await createConversation(variables);
        isCreated = true;
        dismissCreatingToast();
        try {
          if (sessionStorage.getItem(HOME_PROMPT_DRAFT_KEY) === message)
            sessionStorage.removeItem(HOME_PROMPT_DRAFT_KEY);
        } catch {
          // sessionStorage not available
        }
        const targetConversationId = data.conversation_id;
        createdTarget = targetConversationId;
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
              throw new Error(
                t(I18nKey.CONVERSATION$ERROR_STARTING_CONVERSATION),
              );
            }

            setPendingTaskAttachments(taskId, {
              backendId: backend.id,
              orgId: orgId ?? null,
              content: trimmed,
              images: attachmentSnapshot.images,
              files: attachmentSnapshot.files,
              imagesMarkedUploadAsFile: [...imagesMarkedUploadAsFile],
            });
            if (ownsLaunch()) clearAllFiles();
            await enqueueHomeTaskPendingMessage({
              conversationId: targetConversationId,
              text: trimmed,
              images: attachmentSnapshot.images,
              imagesMarkedUploadAsFile,
            });
            if (clientLaunchId) {
              openAutomationSetupMode(targetConversationId);
              useConversationLaunchStore
                .getState()
                .update(clientLaunchId, { targetId: targetConversationId });
              return;
            }
            if (!ownsLaunch()) return;
            openAutomationSetupMode(targetConversationId);
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

        if (clientLaunchId) {
          openAutomationSetupMode(targetConversationId);
          useConversationLaunchStore
            .getState()
            .update(clientLaunchId, { targetId: targetConversationId });
          return;
        }
        if (!ownsLaunch()) return;
        openAutomationSetupMode(targetConversationId);
        navigate(`/conversations/${targetConversationId}`);
      } catch (error) {
        dismissCreatingToast({ immediately: true });
        // Prefer the server's own message; without one, keep the launcher's
        // existing wording (the error's message).
        const fallback = error instanceof Error ? error.message : "";
        const errorMessage =
          getApiErrorMessage(error, fallback) ||
          t(I18nKey.CONVERSATION$ERROR_STARTING_CONVERSATION);
        if (isCreated && clientLaunchId && createdTarget) {
          useConversationLaunchStore
            .getState()
            .update(clientLaunchId, { targetId: createdTarget });
          if (ownsLaunch()) displayErrorToast(errorMessage);
          return;
        }
        if (clientLaunchId)
          useConversationLaunchStore
            .getState()
            .update(clientLaunchId, { error: errorMessage });
        if (!ownsLaunch()) return;
        if (!isLocal) return;
        displayErrorToast(errorMessage);
        // The composer cleared itself on submit; hand the prompt back so the
        // user can retry without retyping it. Only when the create itself
        // failed and this composer is still on screen: the request is consumed
        // by whichever composer is mounted, so otherwise the prompt would
        // replay into a conversation.
        if (!isCreated && isMountedRef.current) {
          restoreMessageToInputIfEmpty(message);
        }
      } finally {
        launchInFlight.current = false;
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
      className={cn(
        "flex w-full flex-col items-center",
        activeLaunch ? "h-full min-h-0 py-4" : "pt-[max(4rem,28vh)] pb-10",
      )}
    >
      <div
        className={cn(
          "flex w-full max-w-200 flex-col gap-4 md:px-4",
          activeLaunch && "min-h-0 flex-1",
        )}
      >
        {activeLaunch && (
          <div className="custom-scrollbar-always flex min-h-0 flex-1 flex-col overflow-y-auto px-4">
            <div
              data-testid="startup-user-message"
              className="mt-6 w-fit max-w-full self-end whitespace-pre-wrap break-words rounded-xl bg-tertiary px-4 py-2.5 text-sm leading-6"
            >
              {activeLaunch.message}
              {activeLaunch.files.length > 0 && (
                <ul>
                  {activeLaunch.files.map((file, index) => (
                    <li key={`${index}-${file.name}`}>{file.name}</li>
                  ))}
                </ul>
              )}
            </div>
            <ConversationStartupProgress
              error={activeLaunch.error}
              onEdit={() => {
                useConversationLaunchStore.getState().remove(activeLaunch.id);
                setPendingLaunchId(null);
                restoreMessageToInputIfEmpty(activeLaunch.message);
                if (activeLaunch.attachments) {
                  const composer = useConversationStore.getState();
                  composer.addImages(activeLaunch.attachments.images);
                  composer.addFiles(activeLaunch.attachments.files);
                  for (const name of activeLaunch.attachments
                    .imagesMarkedUploadAsFile) {
                    if (!composer.imagesMarkedUploadAsFile.includes(name))
                      composer.toggleImageUploadAsFile(name);
                  }
                }
              }}
            />
          </div>
        )}
        <div
          className={cn("flex w-full justify-center", activeLaunch && "hidden")}
        >
          <HomeHeaderTitle />
        </div>

        <div
          className={cn("flex w-full justify-center", activeLaunch && "hidden")}
        >
          <HomeLauncherModeToggle
            mode={displayedLauncherMode}
            onChange={handleLauncherModeChange}
          />
        </div>

        <div className="w-full">
          <CustomChatInput
            onSubmit={handleSubmitWithModelGuard}
            onFilesPaste={handleUpload}
            disabled={isCreating || llmBlocked || !!activeLaunch}
            placeholder={
              isAutomateMode
                ? t(I18nKey.HOME$AUTOMATE_PROMPT_PLACEHOLDER)
                : t(I18nKey.SUGGESTIONS$WHAT_TO_BUILD)
            }
          />
        </div>

        {!activeLaunch && isolated && (
          <p role="status" className="text-xs text-text-secondary">
            {pendingWorkspace
              ? runtimeWorkspaceMessage
              : t(I18nKey.HOME$ISOLATED_WORKSPACE_NEW)}
            {pendingWorkspace && (
              <button
                type="button"
                className="ml-2 underline"
                onClick={() => setPendingWorkspace(null)}
              >
                {t(I18nKey.HOME$CLEAR_HOST_WORKSPACE)}
              </button>
            )}
          </p>
        )}
        <div
          data-testid="home-composer-actions"
          aria-disabled={isAutomateMode}
          className={cn(
            "flex items-center justify-start gap-2",
            activeLaunch && "hidden",
            isAutomateMode && "pointer-events-none opacity-40",
          )}
        >
          {hasSelection ? (
            <HomeGitControlBarPreview
              workspace={pendingWorkspace}
              repository={repositorySelection?.repository}
              branch={repositorySelection?.branch}
              provider={repositorySelection?.provider}
              workspaceMode={effectiveWorkspaceMode}
              workspaceModes={workspaceModes}
              backendKind={backend.kind}
              onRepoClick={() => setIsDialogOpen(true)}
              onClear={
                isLocal
                  ? () => setPendingWorkspace(null)
                  : clearRepositorySelection
              }
              onWorkspaceModeChange={setWorkspaceMode}
            />
          ) : (
            <OpenLauncherButton
              kind={isLocal ? "local" : "cloud"}
              onClick={() => setIsDialogOpen(true)}
              disabled={
                isCreating ||
                isAutomateMode ||
                Boolean(workspacesUnsupportedMessage)
              }
              disabledTooltip={workspacesUnsupportedMessage}
            />
          )}
          {isLocal && !hasSelection && workspaceModes.length > 0 ? (
            <WorkspaceModeSelector
              value={effectiveWorkspaceMode}
              options={workspaceModes}
              backendKind={backend.kind}
              onChange={setWorkspaceMode}
            />
          ) : null}
          <PluginPickerTrigger
            count={selectedPlugins.length}
            onClick={() => setIsPluginPickerOpen(true)}
            disabled={isCreating || isAutomateMode}
          />
        </div>

        <div
          className={cn(
            "mt-8 flex w-full flex-col gap-8",
            activeLaunch && "hidden",
          )}
        >
          {/* @spec HL-002 — Only Home's rail is dismissible; dismissal unmounts it, inset included */}
          {!isRecommendedAutomationsDismissed && (
            <RecommendedAutomationsLauncher
              variant="rail"
              dismissible
              onDismiss={() => writeHomeRecommendedAutomationsDismissed(true)}
            />
          )}
          <PinnedAutomationsDashboard />
          <RunningAutomationsList />
        </div>
      </div>

      {isLocal ? (
        <OpenWorkspaceDialog
          isOpen={isDialogOpen}
          onClose={() => setIsDialogOpen(false)}
          onConfirm={setPendingWorkspace}
        />
      ) : (
        <OpenRepositoryDialog
          isOpen={isDialogOpen}
          onClose={() => setIsDialogOpen(false)}
          onConfirm={({ repository, branch, provider }) => {
            setRepositorySelection({
              repository,
              branch,
              provider: provider ?? repository.git_provider,
            });
            setPickedWorkspaceMode("local_repo");
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
