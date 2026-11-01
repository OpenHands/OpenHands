import { X } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { BackendKind } from "#/api/backend-registry/types";
import type { WorkspaceMode } from "#/api/conversation-metadata-store";
import { I18nKey } from "#/i18n/declaration";
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
  workspaceModes: WorkspaceMode[];
  backendKind: BackendKind;
  onRepoClick: () => void;
  onClear: () => void;
  onWorkspaceModeChange: (mode: WorkspaceMode) => void;
}

export function HomeGitControlBarPreview({
  workspace,
  repository,
  branch,
  provider,
  workspaceMode,
  workspaceModes,
  backendKind,
  onRepoClick,
  onClear,
  onWorkspaceModeChange,
}: HomeGitControlBarPreviewProps) {
  const { t } = useTranslation("openhands");
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
          options={workspaceModes}
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
      <button
        type="button"
        data-testid="home-clear-selection"
        onClick={onClear}
        aria-label={t(I18nKey.COMMON$CLEAR_SELECTION)}
        title={t(I18nKey.COMMON$CLEAR_SELECTION)}
        className="flex h-6 w-6 shrink-0 cursor-pointer items-center justify-center rounded-full text-text-secondary hover:bg-surface-raised hover:text-contrast"
      >
        <X aria-hidden className="h-3.5 w-3.5" strokeWidth={2} />
      </button>
    </div>
  );
}
