import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";

import { useActiveBackend } from "#/contexts/active-backend-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useTaskPolling } from "#/hooks/query/use-task-polling";
import { useLocalGitInfo } from "#/hooks/query/use-local-git-info";
import { useUnifiedWebSocketStatus } from "#/hooks/use-unified-websocket-status";
import { useConversationWebSocket } from "#/contexts/conversation-websocket-context";
import { useSendMessage } from "#/hooks/use-send-message";
import { useUpdateConversationRepository } from "#/hooks/mutation/use-update-conversation-repository";
import { useHomeStore } from "#/stores/home-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { getStoredConversationMetadata } from "#/api/conversation-metadata-store";
import { GitControlBar } from "#/components/features/chat/git-control-bar";
import { ScrollProvider } from "#/context/scroll-context";

// Holder so the OpenRepositoryModal mock can hand its `onLaunch` prop back
// to the test — the modal renders null but the parent's launch flow is the
// only way to drive handleLaunchRepository.
const mocks = vi.hoisted(() => ({
  modalLaunchHandler: {
    current: null as ((repo: unknown, branch: unknown) => void) | null,
  },
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "test-conversation-id" }),
  useConversationId: () => ({ conversationId: "test-conversation-id" }),
}));
vi.mock("#/contexts/active-backend-context");
vi.mock("#/hooks/query/use-active-conversation");
vi.mock("#/hooks/query/use-task-polling");
vi.mock("#/hooks/query/use-local-git-info");
vi.mock("#/hooks/use-unified-websocket-status");
vi.mock("#/contexts/conversation-websocket-context");
vi.mock("#/hooks/use-send-message");
vi.mock("#/hooks/mutation/use-update-conversation-repository");
vi.mock("#/stores/home-store");
vi.mock("#/stores/optimistic-user-message-store");
vi.mock("#/api/conversation-metadata-store");

vi.mock("#/components/features/chat/git-control-bar-repo-button", () => ({
  GitControlBarRepoButton: ({ disabled }: { disabled?: boolean }) => (
    <button
      data-testid="git-control-bar-repo-button"
      type="button"
      data-disabled={String(!!disabled)}
    />
  ),
}));
vi.mock("#/components/features/chat/git-control-bar-branch-button", () => ({
  GitControlBarBranchButton: () => null,
}));
vi.mock("#/components/features/chat/git-control-bar-pull-button", () => ({
  GitControlBarPullButton: () => null,
}));
vi.mock("#/components/features/chat/git-control-bar-push-button", () => ({
  GitControlBarPushButton: () => null,
}));
vi.mock("#/components/features/chat/git-control-bar-pr-button", () => ({
  GitControlBarPrButton: () => null,
}));
vi.mock("#/components/features/chat/git-control-bar-tooltip-wrapper", () => ({
  GitControlBarTooltipWrapper: ({ children }: { children: React.ReactNode }) =>
    children,
}));
vi.mock("#/components/features/chat/open-repository-modal", () => ({
  OpenRepositoryModal: (props: {
    onLaunch?: (repo: unknown, branch: unknown) => void;
  }) => {
    mocks.modalLaunchHandler.current = props.onLaunch ?? null;
    return null;
  },
}));

const makeBackend = (kind: "local" | "cloud") => ({
  backend: {
    id: "backend-id",
    name: kind === "local" ? "Local" : "Cloud",
    host: "http://example.test",
    apiKey: "test-key",
    kind,
  },
  orgId: null,
});

describe("GitControlBar clone prompt format", () => {
  // Helper function that mirrors the logic in git-control-bar.tsx
  const generateClonePrompt = (
    fullName: string,
    gitProvider: string,
    branchName: string,
  ) => {
    const providerName =
      gitProvider.charAt(0).toUpperCase() + gitProvider.slice(1);
    return `Clone ${fullName} from ${providerName} and checkout branch ${branchName}.`;
  };

  it("should include GitHub in clone prompt for github provider", () => {
    const prompt = generateClonePrompt("user/repo", "github", "main");
    expect(prompt).toBe(
      "Clone user/repo from Github and checkout branch main.",
    );
  });

  it("should include GitLab in clone prompt for gitlab provider", () => {
    const prompt = generateClonePrompt("group/project", "gitlab", "develop");
    expect(prompt).toBe(
      "Clone group/project from Gitlab and checkout branch develop.",
    );
  });

  it("should handle different branch names", () => {
    const prompt = generateClonePrompt(
      "hieptl.developer-group/hieptl.developer-project",
      "gitlab",
      "add-batman-microagent",
    );
    expect(prompt).toBe(
      "Clone hieptl.developer-group/hieptl.developer-project from Gitlab and checkout branch add-batman-microagent.",
    );
  });

  it("should capitalize first letter of provider name", () => {
    const githubPrompt = generateClonePrompt("a/b", "github", "main");
    const gitlabPrompt = generateClonePrompt("a/b", "gitlab", "main");

    expect(githubPrompt).toContain("from Github");
    expect(gitlabPrompt).toContain("from Gitlab");
  });
});

describe("GitControlBar repo button visibility", () => {
  beforeEach(() => {
    vi.mocked(useActiveConversation).mockReturnValue({
      data: { id: "test-conversation-id" },
    } as ReturnType<typeof useActiveConversation>);
    vi.mocked(useTaskPolling).mockReturnValue({
      repositoryInfo: null,
    } as unknown as ReturnType<typeof useTaskPolling>);
    vi.mocked(useLocalGitInfo).mockReturnValue({
      data: null,
    } as unknown as ReturnType<typeof useLocalGitInfo>);
    vi.mocked(useUnifiedWebSocketStatus).mockReturnValue("OPEN");
    vi.mocked(useConversationWebSocket).mockReturnValue({
      isLoadingHistory: false,
    } as ReturnType<typeof useConversationWebSocket>);
    vi.mocked(useSendMessage).mockReturnValue({
      send: vi.fn(),
    } as unknown as ReturnType<typeof useSendMessage>);
    vi.mocked(useUpdateConversationRepository).mockReturnValue({
      mutate: vi.fn(),
    } as unknown as ReturnType<typeof useUpdateConversationRepository>);
    vi.mocked(useHomeStore).mockReturnValue({
      addRecentRepository: vi.fn(),
    } as unknown as ReturnType<typeof useHomeStore>);
    vi.mocked(useOptimisticUserMessageStore).mockImplementation(((
      selector: (s: unknown) => unknown,
    ) =>
      selector({
        enqueuePendingMessage: vi.fn(),
        markPendingMessageError: vi.fn(),
      })) as unknown as typeof useOptimisticUserMessageStore);
    vi.mocked(getStoredConversationMetadata).mockReturnValue(null);
  });

  it("hides the repo button on a local backend with no repository or workspace name", () => {
    vi.mocked(useActiveBackend).mockReturnValue(makeBackend("local"));

    renderWithProviders(<GitControlBar onSuggestionsClick={vi.fn()} />);

    expect(
      screen.queryByTestId("git-control-bar-repo-button"),
    ).not.toBeInTheDocument();
  });

  it("shows the repo button on a cloud backend with no repository connected", () => {
    vi.mocked(useActiveBackend).mockReturnValue(makeBackend("cloud"));

    renderWithProviders(<GitControlBar onSuggestionsClick={vi.fn()} />);

    expect(
      screen.getByTestId("git-control-bar-repo-button"),
    ).toBeInTheDocument();
  });

  it("renders the repo button as disabled on a local backend when only a workspace name is available", () => {
    vi.mocked(useActiveBackend).mockReturnValue(makeBackend("local"));
    vi.mocked(getStoredConversationMetadata).mockReturnValue({
      selected_workspace: "/projects/my-app",
    } as ReturnType<typeof getStoredConversationMetadata>);

    renderWithProviders(<GitControlBar onSuggestionsClick={vi.fn()} />);

    const button = screen.getByTestId("git-control-bar-repo-button");
    expect(button).toHaveAttribute("data-disabled", "true");
  });

  it("renders the repo button as disabled while conversation history is loading", () => {
    vi.mocked(useActiveBackend).mockReturnValue(makeBackend("cloud"));
    vi.mocked(useConversationWebSocket).mockReturnValue({
      isLoadingHistory: true,
    } as ReturnType<typeof useConversationWebSocket>);

    renderWithProviders(<GitControlBar onSuggestionsClick={vi.fn()} />);

    const button = screen.getByTestId("git-control-bar-repo-button");
    expect(button).toHaveAttribute("data-disabled", "true");
  });
});

describe("GitControlBar - Auto-scroll on clone (issue #817)", () => {
  beforeEach(() => {
    mocks.modalLaunchHandler.current = null;
    vi.mocked(useActiveBackend).mockReturnValue(makeBackend("cloud"));
    vi.mocked(useActiveConversation).mockReturnValue({
      data: { id: "test-conversation-id" },
    } as ReturnType<typeof useActiveConversation>);
    vi.mocked(useTaskPolling).mockReturnValue({
      repositoryInfo: null,
    } as unknown as ReturnType<typeof useTaskPolling>);
    vi.mocked(useLocalGitInfo).mockReturnValue({
      data: null,
    } as unknown as ReturnType<typeof useLocalGitInfo>);
    vi.mocked(useUnifiedWebSocketStatus).mockReturnValue("OPEN");
    vi.mocked(useSendMessage).mockReturnValue({
      send: vi.fn().mockResolvedValue(undefined),
    } as unknown as ReturnType<typeof useSendMessage>);
    // updateRepository mocked to invoke onSuccess synchronously so the
    // post-update branch (where the scroll now lives) runs deterministically.
    vi.mocked(useUpdateConversationRepository).mockReturnValue({
      mutate: (_args: unknown, options: { onSuccess?: () => void }) => {
        options?.onSuccess?.();
      },
    } as unknown as ReturnType<typeof useUpdateConversationRepository>);
    vi.mocked(useHomeStore).mockReturnValue({
      addRecentRepository: vi.fn(),
    } as unknown as ReturnType<typeof useHomeStore>);
    vi.mocked(useOptimisticUserMessageStore).mockImplementation(((
      selector: (s: unknown) => unknown,
    ) =>
      selector({
        enqueuePendingMessage: vi.fn().mockReturnValue("pending-id"),
        markPendingMessageError: vi.fn(),
      })) as unknown as typeof useOptimisticUserMessageStore);
    vi.mocked(getStoredConversationMetadata).mockReturnValue(null);
  });

  it("scrolls the chat to bottom after a successful clone is enqueued", () => {
    // Arrange: provide a spy via ScrollProvider so we can observe the
    // scroll callback the bar pulls out of useOptionalScrollContext.
    const scrollDomToBottom = vi.fn();
    const scrollValue = {
      scrollRef: { current: null },
      autoScroll: true,
      setAutoScroll: vi.fn(),
      scrollDomToBottom,
      hitBottom: true,
      setHitBottom: vi.fn(),
      onChatBodyScroll: vi.fn(),
    };

    renderWithProviders(
      <ScrollProvider value={scrollValue}>
        <GitControlBar onSuggestionsClick={vi.fn()} />
      </ScrollProvider>,
    );

    expect(typeof mocks.modalLaunchHandler.current).toBe("function");

    // Act: drive handleLaunchRepository the same way the modal does.
    mocks.modalLaunchHandler.current?.(
      { full_name: "user/repo", git_provider: "github" },
      { name: "main" },
    );

    // Assert: the optimistic clone bubble must pull the chat back to the
    // bottom even if the user had scrolled up.
    expect(scrollDomToBottom).toHaveBeenCalledTimes(1);
  });
});
