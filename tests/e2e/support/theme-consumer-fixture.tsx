/* eslint-disable i18next/no-literal-string -- test-only fixture labels identify the exercised control states */
import { createPortal } from "react-dom";
import { AgentServerUIRoot } from "../../../src/components/providers/agent-server-ui-root";
import { CirclePlusCheckToggle } from "../../../src/components/shared/buttons/circle-plus-check-toggle";
import { SettingsInput } from "../../../src/components/features/settings/settings-input";
import { UploadedFile } from "../../../src/components/features/chat/uploaded-file";
import { createRoot } from "react-dom/client";
import { AgentServerUIProviders } from "../../../src/components/providers/agent-server-ui-providers";
import { ActiveStatusBadge } from "../../../src/components/features/automations/detail/active-status-badge";
import { GitControlBarBranchButton } from "../../../src/components/features/chat/git-control-bar-branch-button";
import { GitControlBarRepoButton } from "../../../src/components/features/chat/git-control-bar-repo-button";
import { WorkspaceModeSelector } from "../../../src/components/features/chat/workspace-mode-selector";
import { getWorkspaceModeOptions } from "../../../src/utils/workspace-mode";

export function mountThemeConsumerFixture(
  scope: Element,
  appearanceOverride?: "light" | "dark",
) {
  // A sibling root has no inherited appearance class from the standalone app.
  const embeddedContainer = appearanceOverride
    ? document.createElement("section")
    : null;
  if (embeddedContainer) {
    embeddedContainer.style.cssText =
      "position:fixed;top:16px;right:16px;z-index:9999;width:240px";
    document.body.append(embeddedContainer);
  }
  const container = document.createElement("section");
  container.dataset.testid = "theme-consumer-fixture";
  container.style.cssText =
    "position:fixed;bottom:16px;left:16px;z-index:9999;display:flex;flex-wrap:wrap;align-items:center;max-width:calc(100vw - 32px);gap:16px;padding:16px;background:var(--oh-surface)";
  scope.append(container);
  const root = createRoot(container);
  root.render(
    <AgentServerUIProviders withStyleRoot={false} analytics={false}>
      {embeddedContainer &&
        createPortal(
          <AgentServerUIRoot
            theme={appearanceOverride}
            styleOverrides={{
              "--oh-border": "#853b9a",
              "--oh-border-input": "#197256",
            }}
          >
            <SettingsInput
              type="text"
              testId="theme-appearance-override"
              name="theme-appearance-override"
              label="Embedded appearance"
              defaultValue="Host override"
            />
          </AgentServerUIRoot>,
          embeddedContainer,
        )}
      <div className="w-64 flex flex-col gap-2">
        <SettingsInput
          type="text"
          testId="theme-caller-invalid"
          name="theme-caller-invalid"
          label="Caller validation"
          defaultValue="invalid name!"
          ariaInvalid
          inputClassName="border-danger"
        />
        <SettingsInput
          type="text"
          testId="theme-error"
          name="theme-error"
          label="Validation error"
          defaultValue="invalid name!"
          error="Use a valid name"
        />
        <SettingsInput
          type="text"
          testId="theme-disabled"
          name="theme-disabled"
          label="Disabled control"
          defaultValue="Read only while saving"
          isDisabled
        />
      </div>
      <div data-testid="theme-uploaded-file">
        <UploadedFile
          file={
            new File(["theme sample"], "sample.txt", { type: "text/plain" })
          }
          onRemove={() => {}}
        />
      </div>
      <CirclePlusCheckToggle
        testId="theme-selected-toggle"
        isSelected
        onToggle={() => {}}
      />
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
    embeddedContainer?.remove();
  };
}
