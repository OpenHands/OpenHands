import { createRoot } from "react-dom/client";
import { AgentServerUIProviders } from "../../../src/components/providers/agent-server-ui-providers";
import { ActiveStatusBadge } from "../../../src/components/features/automations/detail/active-status-badge";
import { GitControlBarBranchButton } from "../../../src/components/features/chat/git-control-bar-branch-button";
import { GitControlBarRepoButton } from "../../../src/components/features/chat/git-control-bar-repo-button";
import { WorkspaceModeSelector } from "../../../src/components/features/chat/workspace-mode-selector";
import { getWorkspaceModeOptions } from "../../../src/utils/workspace-mode";

export function mountThemeConsumerFixture(scope: Element) {
  const container = document.createElement("section");
  container.dataset.testid = "theme-consumer-fixture";
  container.style.cssText =
    "position:fixed;bottom:16px;left:16px;z-index:9999;display:flex;gap:16px;padding:16px;background:var(--oh-surface)";
  scope.append(container);
  const root = createRoot(container);
  root.render(
    <AgentServerUIProviders withStyleRoot={false} analytics={false}>
      <ActiveStatusBadge automation={{ enabled: false, state: "DRAFT" }} />
      <div data-testid="theme-branch-icon">
        <GitControlBarBranchButton
          selectedBranch="main"
          selectedRepository="qa/repo"
          gitProvider="github"
        />
      </div>
      <div data-testid="theme-repo-icon">
        <GitControlBarRepoButton
          selectedRepository={null}
          gitProvider={null}
          workspaceName="local-repo"
        />
      </div>
      <WorkspaceModeSelector
        value="local_repo"
        options={getWorkspaceModeOptions(false, true)}
        backendKind="local"
        onChange={() => {}}
      />
    </AgentServerUIProviders>,
  );
  return () => {
    root.unmount();
    container.remove();
  };
}
