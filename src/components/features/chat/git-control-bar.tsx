import { useState, useRef, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { GitControlBarRepoButton } from "./git-control-bar-repo-button";
import { GitControlBarBranchButton } from "./git-control-bar-branch-button";
import { GitControlBarPullButton } from "./git-control-bar-pull-button";
import { GitControlBarPushButton } from "./git-control-bar-push-button";
import { GitControlBarPrButton } from "./git-control-bar-pr-button";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useLocalGitInfo } from "#/hooks/query/use-local-git-info";
import { useTaskPolling } from "#/hooks/query/use-task-polling";
import { useUnifiedWebSocketStatus } from "#/hooks/use-unified-websocket-status";
import { useConversationWebSocket } from "#/contexts/conversation-websocket-context";
import { useSendMessage } from "#/hooks/use-send-message";
import { useUpdateConversationRepository } from "#/hooks/mutation/use-update-conversation-repository";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { Provider } from "#/types/settings";
import { Branch, GitRepository } from "#/types/git";
import { I18nKey } from "#/i18n/declaration";
import { GitControlBarTooltipWrapper } from "./git-control-bar-tooltip-wrapper";
import { OpenRepositoryModal } from "./open-repository-modal";
import { useConversationId } from "#/hooks/use-conversation-id";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { useHomeStore } from "#/stores/home-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { getStoredConversationMetadata } from "#/api/conversation-metadata-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useUserProviders } from "#/hooks/use-user-providers";
import { useOptionalScrollContext } from "#/context/scroll-context";

interface GitControlBarProps {
  onSuggestionsClick: (value: string) => void;
}

export function GitControlBar({ onSuggestionsClick }: GitControlBarProps) {
  const { t } = useTranslation("openhands");
  const { conversationId } = useConversationId();
  const [isOpenRepoModalOpen, setIsOpenRepoModalOpen] = useState(false);
  const [isWorkspaceMenuOpen, setIsWorkspaceMenuOpen] = useState(false);
  const workspaceMenuContainerRef = useRef<HTMLDivElement>(null);
  const { addRecentRepository } = useHomeStore();
  const enqueuePendingMessage = useOptimisticUserMessageStore(
    (state) => state.enqueuePendingMessage,
  );
  const markPendingMessageError = useOptimisticUserMessageStore(
    (state) => state.markPendingMessageError,
  );
  const { backend } = useActiveBackend();
  const isLocalBackend = backend.kind === "local";
  const { providers } = useUserProviders();
  const providerTokensReady = isLocalBackend || providers.length > 0;

  const { data: conversation } = useActiveConversation();
  const { repositoryInfo } = useTaskPolling();
  const { data: localGitInfo } = useLocalGitInfo();
  const webSocketStatus = useUnifiedWebSocketStatus();
  const conversationWebSocket = useConversationWebSocket();
  const isLoadingHistory = conversationWebSocket?.isLoadingHistory ?? false;
  const webSocketStatusRef = useRef(webSocketStatus);
  useEffect(() => {
    webSocketStatusRef.current = webSocketStatus;
  }, [webSocketStatus]);
  const { send } = useSendMessage();
  const sendRef = useRef(send);
  useEffect(() => {
    sendRef.current = send;
  }, [send]);
  const scrollContext = useOptionalScrollContext();
  const { mutate: updateRepository } = useUpdateConversationRepository();
  const { mutate: _createConversation, isPending: _isCreatingConversation } =
    useCreateConversation();

  // Priority: conversation data > task data > locally-detected git info.
  // The local fallback runs `git remote get-url origin` / `git rev-parse --abbrev-ref HEAD`
  // in the conversation's working dir so local-workspace conversations can
  // still display a repo and branch in the control bar.
  const conversationRepository =
    conversation?.selected_repository || repositoryInfo?.selectedRepository;
  const conversationProvider = (conversation?.git_provider ||
    repositoryInfo?.gitProvider) as Provider | undefined;
  const conversationBranch =
    conversation?.selected_branch || repositoryInfo?.selectedBranch;

  const selectedRepository =
    conversationRepository || localGitInfo?.repository || undefined;
  const gitProvider = (conversationProvider ||
    localGitInfo?.provider) as Provider;
  const selectedBranch =
    conversationBranch || localGitInfo?.branch || undefined;

  // For folder-only conversations (no remote repo), surface the basename of
  // the originally attached workspace path so the button reads e.g. "test"
  // rather than "No Repo Connected". `selected_workspace` is recorded at
  // conversation creation; we prefer it over `workspace.working_dir` because
  // the latter may point at a worktree subdir.
  const storedMetadata = conversation?.id
    ? getStoredConversationMetadata(conversation.id)
    : null;
  const workspacePath = storedMetadata?.selected_workspace ?? null;
  const workspaceName = workspacePath
    ? workspacePath.replace(/\/+$/, "").split("/").pop() || null
    : null;

  // Enable git actions whenever a repository and provider are known, including
  // local conversations where repo metadata is inferred from git remotes.
  const hasRepository = !!selectedRepository && !!gitProvider;

  // Enable buttons only when conversation exists, WS is connected, and the
  // initial history preload has finished (matches chat-interface loading gate).
  const isConversationReady =
    !!conversation && webSocketStatus === "OPEN" && !isLoadingHistory;

  useEffect(() => {
    if (!isWorkspaceMenuOpen) return undefined;
    const onMouseDown = (event: MouseEvent) => {
      if (
        workspaceMenuContainerRef.current &&
        !workspaceMenuContainerRef.current.contains(event.target as Node)
      ) {
        setIsWorkspaceMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, [isWorkspaceMenuOpen]);

  useEffect(() => {
    if (!isWorkspaceMenuOpen) return undefined;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsWorkspaceMenuOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isWorkspaceMenuOpen]);

  const handleLaunchRepository = (
    repository: GitRepository,
    branch: Branch,
  ) => {
    if (!conversationId) return;

    // Persist to recent repositories list (matches home page behavior)
    addRecentRepository(repository);

    // Note: We update repository metadata first, then send clone command.
    // The clone command is sent to the agent via WebSocket (fire-and-forget).
    // If cloning fails, the agent will report the error in the chat,
    // and the user can retry or change the repository.
    // This is a trade-off: immediate UI feedback vs. strict atomicity.
    updateRepository(
      {
        conversationId,
        repository: repository.full_name,
        branch: branch.name,
        gitProvider: repository.git_provider,
      },
      {
        onSuccess: () => {
          // Use ref to read the latest WebSocket status (avoids stale closure)
          if (webSocketStatusRef.current !== "OPEN") {
            displayErrorToast(
              t(I18nKey.CONVERSATION$CLONE_COMMAND_FAILED_DISCONNECTED),
            );
            return;
          }

          // Send clone command to agent after metadata is updated
          // Use ref to always call the latest send function (avoids stale closure
          // where V1 sendMessage holds a reference to a now-closed WebSocket)
          // Include git provider in prompt so agent clones from correct source
          const providerName =
            repository.git_provider.charAt(0).toUpperCase() +
            repository.git_provider.slice(1);
          const clonePrompt = `Clone ${repository.full_name} from ${providerName} and checkout branch ${branch.name}.`;
          const pendingId = conversationId
            ? enqueuePendingMessage({ conversationId, text: clonePrompt })
            : null;
          // Pull chat back to the bottom so the optimistic "Clone …" bubble
          // is visible even if the user had scrolled up.
          scrollContext?.scrollDomToBottom();
          // `send` returns a Promise; surface a failed send by flipping the
          // matching pending entry to "error" so the user gets the retry link
          // rather than a perpetual "Sending…" bubble.
          Promise.resolve(
            sendRef.current({
              action: "message",
              args: {
                content: clonePrompt,
                timestamp: new Date().toISOString(),
              },
            }),
          ).catch((error) => {
            if (!pendingId) return;
            const errorMessage =
              error instanceof Error
                ? error.message
                : t(I18nKey.CHAT_INTERFACE$FAILED_TO_SEND_MESSAGE);
            markPendingMessageError(pendingId, errorMessage);
          });
        },
      },
    );
  };

  // Local backends never use the remote-repo "Connect Repo" CTA, so suppress the
  // empty-state button there. A repo or workspace label inferred from local git
  // metadata is still informational and stays visible.
  const showRepoButton =
    !isLocalBackend || !!selectedRepository || !!workspaceName;
  // On a local backend the informational pill (e.g. workspace name, or a repo
  // detected without a recognized provider) should not open the remote-repo
  // modal — that flow is cloud-only. Disable the button in that case so the
  // click is a no-op. Linkable repos render as <a> and ignore `disabled`.
  const isRepoButtonInert = isLocalBackend && !hasRepository;

  // True when the bar will render at least one chip (cloud always shows
  // "Open Repository"; local needs a repo or a workspace name; selected
  // branch or push/pull/PR also count). When false, the bar has nothing to
  // show — return null so the wrapper above collapses to its natural padding
  // instead of leaving an empty DOM node below the chat input.
  const hasAnyContent = showRepoButton || !!selectedBranch || hasRepository;
  if (!hasAnyContent) return null;

  return (
    <div className="flex flex-row items-center">
      <div className="flex flex-row gap-2.5 items-center overflow-x-auto flex-nowrap relative scrollbar-hide">
        {showRepoButton ? (
          <GitControlBarRepoButton
            selectedRepository={selectedRepository}
            gitProvider={gitProvider}
            workspaceName={workspaceName}
            onClick={() => setIsOpenRepoModalOpen(true)}
            disabled={!isConversationReady || isRepoButtonInert}
          />
        ) : null}

        {selectedBranch ? (
          <GitControlBarBranchButton
            selectedBranch={selectedBranch}
            selectedRepository={selectedRepository}
            gitProvider={gitProvider}
          />
        ) : null}

        {hasRepository ? (
          <>
            <GitControlBarTooltipWrapper
              tooltipMessage={t(I18nKey.COMMON$GIT_TOOLS_DISABLED_CONTENT)}
              testId="git-control-bar-pull-button-tooltip"
              shouldShowTooltip={!hasRepository}
            >
              <GitControlBarPullButton
                onSuggestionsClick={onSuggestionsClick}
                hasRepository={hasRepository}
                providerTokensReady={providerTokensReady}
                isConversationReady={isConversationReady}
              />
            </GitControlBarTooltipWrapper>

            <GitControlBarTooltipWrapper
              tooltipMessage={t(I18nKey.COMMON$GIT_TOOLS_DISABLED_CONTENT)}
              testId="git-control-bar-push-button-tooltip"
              shouldShowTooltip={!hasRepository}
            >
              <GitControlBarPushButton
                onSuggestionsClick={onSuggestionsClick}
                hasRepository={hasRepository}
                providerTokensReady={providerTokensReady}
                currentGitProvider={gitProvider}
                isConversationReady={isConversationReady}
              />
            </GitControlBarTooltipWrapper>

            <GitControlBarTooltipWrapper
              tooltipMessage={t(I18nKey.COMMON$GIT_TOOLS_DISABLED_CONTENT)}
              testId="git-control-bar-pr-button-tooltip"
              shouldShowTooltip={!hasRepository}
            >
              <GitControlBarPrButton
                onSuggestionsClick={onSuggestionsClick}
                hasRepository={hasRepository}
                providerTokensReady={providerTokensReady}
                currentGitProvider={gitProvider}
                isConversationReady={isConversationReady}
              />
            </GitControlBarTooltipWrapper>
          </>
        ) : null}
      </div>

      <OpenRepositoryModal
        isOpen={isOpenRepoModalOpen}
        onClose={() => setIsOpenRepoModalOpen(false)}
        onLaunch={handleLaunchRepository}
        defaultProvider={gitProvider}
      />
    </div>
  );
}
