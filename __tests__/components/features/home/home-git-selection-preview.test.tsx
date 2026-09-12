import type { ComponentProps } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { GitControlBarRepoButton as RepoButtonComponent } from "#/components/features/chat/git-control-bar-repo-button";
import type { GitControlBarBranchButton as BranchButtonComponent } from "#/components/features/chat/git-control-bar-branch-button";
import type { WorkspaceModeSelector as WorkspaceModeSelectorComponent } from "#/components/features/chat/workspace-mode-selector";
import { HomeGitControlBarPreview } from "#/components/features/home/home-git-control-bar-preview";
import type { Branch, GitRepository } from "#/types/git";
import type { LocalWorkspace } from "#/types/workspace";

type RepoButtonProps = ComponentProps<typeof RepoButtonComponent>;
type BranchButtonProps = ComponentProps<typeof BranchButtonComponent>;
type WorkspaceModeSelectorProps = ComponentProps<
  typeof WorkspaceModeSelectorComponent
>;

vi.mock("#/components/features/chat/git-control-bar-repo-button", () => ({
  GitControlBarRepoButton: ({
    selectedRepository,
    gitProvider,
    workspaceName,
    onClick,
  }: RepoButtonProps) => (
    <button type="button" data-testid="repo-preview" onClick={onClick}>
      <span data-testid="repo-name">{selectedRepository ?? "none"}</span>
      <span data-testid="repo-provider">{gitProvider ?? "none"}</span>
      <span data-testid="workspace-name">{workspaceName ?? "none"}</span>
    </button>
  ),
}));

vi.mock("#/components/features/chat/git-control-bar-branch-button", () => ({
  GitControlBarBranchButton: ({
    selectedBranch,
    selectedRepository,
    gitProvider,
  }: BranchButtonProps) => (
    <div data-testid="branch-preview">
      <span data-testid="branch-name">{selectedBranch ?? "none"}</span>
      <span data-testid="branch-repository">
        {selectedRepository ?? "none"}
      </span>
      <span data-testid="branch-provider">{gitProvider ?? "none"}</span>
    </div>
  ),
}));

vi.mock("#/components/features/chat/workspace-mode-selector", () => ({
  WorkspaceModeSelector: ({
    value,
    backendKind,
    onChange,
  }: WorkspaceModeSelectorProps) => (
    <button
      type="button"
      data-testid="workspace-mode-preview"
      data-value={value}
      data-backend-kind={backendKind}
      onClick={() =>
        onChange(value === "local_repo" ? "new_worktree" : "local_repo")
      }
    >
      Change workspace mode
    </button>
  ),
}));

type PreviewProps = ComponentProps<typeof HomeGitControlBarPreview>;

function createWorkspace(
  overrides: Partial<LocalWorkspace> = {},
): LocalWorkspace {
  return {
    id: "workspace-1",
    name: "Agent Canvas",
    path: "/workspace/agent-canvas",
    ...overrides,
  };
}

function createRepository(
  overrides: Partial<GitRepository> = {},
): GitRepository {
  return {
    id: "repository-1",
    full_name: "OpenHands/agent-canvas",
    git_provider: "github",
    is_public: true,
    ...overrides,
  };
}

function createBranch(overrides: Partial<Branch> = {}): Branch {
  return {
    name: "main",
    commit_sha: "abc123",
    protected: true,
    ...overrides,
  };
}

function createPreviewProps(
  overrides: Partial<PreviewProps> = {},
): PreviewProps {
  return {
    workspace: createWorkspace(),
    repository: createRepository(),
    branch: createBranch(),
    provider: "github",
    workspaceMode: "local_repo",
    backendKind: "local",
    onRepoClick: vi.fn(),
    onWorkspaceModeChange: vi.fn(),
    ...overrides,
  };
}

describe("home git selection preview", () => {
  it("shows a complete selection and forwards both controls", async () => {
    const user = userEvent.setup();
    const props = createPreviewProps({
      workspace: createWorkspace({ path: "/workspace/review-target///" }),
    });

    render(<HomeGitControlBarPreview {...props} />);

    expect(
      screen.getByTestId("home-git-control-bar-preview"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("repo-name")).toHaveTextContent(
      "OpenHands/agent-canvas",
    );
    expect(screen.getByTestId("repo-provider")).toHaveTextContent("github");
    expect(screen.getByTestId("workspace-name")).toHaveTextContent(
      /^review-target$/,
    );
    expect(screen.getByTestId("workspace-mode-preview")).toHaveAttribute(
      "data-value",
      "local_repo",
    );
    expect(screen.getByTestId("workspace-mode-preview")).toHaveAttribute(
      "data-backend-kind",
      "local",
    );
    expect(screen.getByTestId("branch-name")).toHaveTextContent("main");
    expect(screen.getByTestId("branch-repository")).toHaveTextContent(
      "OpenHands/agent-canvas",
    );
    expect(screen.getByTestId("branch-provider")).toHaveTextContent("github");

    await user.click(screen.getByTestId("repo-preview"));
    await user.click(screen.getByTestId("workspace-mode-preview"));

    expect(props.onRepoClick).toHaveBeenCalledOnce();
    expect(props.onWorkspaceModeChange).toHaveBeenCalledWith("new_worktree");
  });

  it("keeps only the repository control when there is no selection", () => {
    render(
      <HomeGitControlBarPreview
        {...createPreviewProps({
          workspace: null,
          repository: null,
          branch: null,
          provider: null,
        })}
      />,
    );

    expect(screen.getByTestId("repo-preview")).toBeInTheDocument();
    expect(screen.getByTestId("repo-name")).toHaveTextContent("none");
    expect(screen.getByTestId("repo-provider")).toHaveTextContent("none");
    expect(screen.getByTestId("workspace-name")).toHaveTextContent("none");
    expect(
      screen.queryByTestId("workspace-mode-preview"),
    ).not.toBeInTheDocument();
    expect(screen.queryByTestId("branch-preview")).not.toBeInTheDocument();
  });

  it("falls back to the full workspace path and shows an unlinked branch", async () => {
    const user = userEvent.setup();
    const props = createPreviewProps({
      workspace: createWorkspace({ path: "/" }),
      repository: null,
      branch: createBranch({ name: "feature/local" }),
      provider: null,
      workspaceMode: "new_worktree",
      backendKind: "cloud",
    });

    render(<HomeGitControlBarPreview {...props} />);

    expect(screen.getByTestId("workspace-name")).toHaveTextContent("/");
    expect(screen.getByTestId("workspace-mode-preview")).toHaveAttribute(
      "data-value",
      "new_worktree",
    );
    expect(screen.getByTestId("workspace-mode-preview")).toHaveAttribute(
      "data-backend-kind",
      "cloud",
    );
    expect(screen.getByTestId("branch-name")).toHaveTextContent(
      "feature/local",
    );
    expect(screen.getByTestId("branch-repository")).toHaveTextContent("none");
    expect(screen.getByTestId("branch-provider")).toHaveTextContent("none");

    await user.click(screen.getByTestId("workspace-mode-preview"));
    expect(props.onWorkspaceModeChange).toHaveBeenCalledWith("local_repo");
  });
});
