import type { BackendKind } from "#/api/backend-registry/types";
import type { WorkspaceMode } from "#/api/conversation-metadata-store";
import { GitControlBarRepoButton } from "#/components/features/chat/git-control-bar-repo-button";
import { GitControlBarBranchButton } from "#/components/features/chat/git-control-bar-branch-button";
import { WorkspaceModeSelector } from "#/components/features/chat/workspace-mode-selector";
import { Branch, GitRepository } from "#/types/git";
import { Provider } from "#/types/settings";
import { LocalWorkspace } from "#/types/workspace";

interface HomeGitControlBarPreviewProps {
  workspace?: LocalWorkspace | null;
  repository?: GitRepository | null;
  branch?: Branch | null;
  provider?: Provider | null;
  workspaceMode: WorkspaceMode;
  backendKind: BackendKind;
  onRepoClick: () => void;
  onWorkspaceModeChange: (mode: WorkspaceMode) => void;
}

export function HomeGitControlBarPreview({
  workspace,
  repository,
  branch,
  provider,
  workspaceMode,
  backendKind,
  onRepoClick,
  onWorkspaceModeChange,
}: HomeGitControlBarPreviewProps) {
  const workspaceName = workspace
    ? workspace.path.replace(/\/+$/, "").split("/").pop() || workspace.path
    : null;

  return (
    <div
      className="flex flex-row gap-2.5 items-center flex-wrap"
      data-testid="home-git-control-bar-preview"
    >
      <GitControlBarRepoButton
        selectedRepository={repository?.full_name ?? null}
        gitProvider={provider ?? null}
        workspaceName={workspaceName}
        onClick={onRepoClick}
      />
      {workspace ? (
        <WorkspaceModeSelector
          value={workspaceMode}
          backendKind={backendKind}
          onChange={onWorkspaceModeChange}
        />
      ) : null}
      {branch ? (
        <GitControlBarBranchButton
          selectedBranch={branch.name}
          selectedRepository={repository?.full_name ?? null}
          gitProvider={provider ?? null}
        />
      ) : null}
    </div>
  );
}
