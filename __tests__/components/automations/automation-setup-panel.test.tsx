import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";
import { AutomationSetupPanel } from "#/components/features/automations/setup/automation-setup-panel";
import AutomationService from "#/api/automation-service/automation-service.api";
import type { AutomationSetupDraft } from "#/api/automation-setup-types";
import { packTarGzip } from "#/utils/tar-gzip";

const mockNavigate = vi.fn();
const mockToastSuccess = vi.fn();

vi.mock("react-hot-toast", () => ({
  default: { success: (...args: unknown[]) => mockToastSuccess(...args) },
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
}));

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    validateDraft: vi.fn(),
    createAutomationDraft: vi.fn(),
    uploadAutomationTarball: vi.fn(),
  },
}));

vi.mock("#/utils/tar-gzip", () => ({
  packTarGzip: vi.fn().mockResolvedValue(new Uint8Array([1, 2, 3])),
}));

vi.mock("#/manifests/automation-interface", () => ({
  automationDetailPath: (id: string) => `/automations/${id}`,
  getAutomationEndpoint: (name: string) =>
    name === "createPlugin"
      ? "/v1/preset/plugin"
      : name === "createBundle"
        ? "/v1"
        : "/v1/preset/prompt",
}));

vi.mock("#/hooks/query/use-llm-profiles", () => ({
  useLlmProfiles: () => ({
    data: { active_profile: "default", profiles: [] },
    isLoading: false,
  }),
}));

vi.mock("#/hooks/query/use-agent-profiles", () => ({
  useAgentProfiles: () => ({
    data: { active_agent_profile_id: null, profiles: [] },
    isLoading: false,
  }),
}));

vi.mock("#/hooks/use-user-providers", () => ({
  useUserProviders: () => ({ providers: [] }),
}));

vi.mock("#/hooks/query/use-git-repositories", () => ({
  useGitRepositories: () => ({ data: { pages: [] }, isLoading: false }),
}));

function renderPanel(
  draft: AutomationSetupDraft = {
    prompt: "Review every pull request",
    kind: "prompt",
  },
) {
  const value: NavigationContextValue = {
    currentPath: "/conversations/conv-1",
    conversationId: "conv-1",
    isNavigating: false,
    navigate: mockNavigate,
  };

  return render(
    <NavigationProvider value={value}>
      <AutomationSetupPanel draft={draft} onClose={vi.fn()} />
    </NavigationProvider>,
  );
}

describe("AutomationSetupPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders compact top toolbar actions", () => {
    renderPanel();

    expect(screen.getByTestId("automation-setup-save-draft")).toHaveClass(
      "!h-7",
      "!min-h-7",
      "!px-2.5",
      "!text-xs",
    );
    expect(screen.getByTestId("automation-setup-test")).toHaveClass("!h-7");
    expect(screen.getByTestId("automation-setup-create")).toHaveClass("!h-7");
  });

  it("switches between prompt, plugin, and custom form types", async () => {
    const user = userEvent.setup();
    renderPanel();

    expect(screen.getByTestId("automation-setup-prompt")).toHaveValue(
      "Review every pull request",
    );
    expect(screen.getByTestId("automation-setup-save-draft")).toBeDisabled();

    await user.click(screen.getByTestId("automation-setup-add-plugin"));
    expect(
      screen.getByTestId("automation-setup-plugin-source"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("automation-setup-kind-custom"));
    expect(screen.getByTestId("automation-setup-entrypoint")).toHaveValue(
      "python3 main.py",
    );
    expect(
      screen.getByTestId("automation-setup-setup-script-path"),
    ).toHaveValue("setup.sh");
    expect(
      (
        screen.getByTestId(
          "automation-setup-custom-code",
        ) as HTMLTextAreaElement
      ).value,
    ).toContain("Review every pull request");
    expect(
      screen.queryByTestId("automation-setup-rendered-code"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("automation-setup-prompt"),
    ).not.toBeInTheDocument();
  });

  it("validates the prompt draft against the automation service", async () => {
    vi.mocked(AutomationService.validateDraft).mockResolvedValue({
      valid: true,
      errors: [],
    });

    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-test"));

    await waitFor(() =>
      expect(AutomationService.validateDraft).toHaveBeenCalledWith({
        endpoint: "/v1/preset/prompt",
        draft: expect.objectContaining({
          enabled: false,
          prompt: "Review every pull request",
          trigger: {
            type: "cron",
            schedule: "0 9 * * *",
            timezone: "America/New_York",
          },
        }),
      }),
    );
    expect(screen.getByTestId("automation-setup-status")).toHaveTextContent(
      "AUTOMATION_SETUP$TEST_PASSED",
    );
  });

  it("sends each comma-separated repository to the automation service", async () => {
    vi.mocked(AutomationService.validateDraft).mockResolvedValue({
      valid: true,
      errors: [],
    });

    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-repository-add"));
    await user.click(screen.getByTestId("automation-setup-repository-custom"));
    await user.type(
      screen.getByTestId("automation-setup-repository-address"),
      "OpenHands/OpenHands, https://gitlab.com/acme/project",
    );
    await user.click(screen.getByTestId("automation-setup-repository-submit"));
    await user.click(screen.getByTestId("automation-setup-test"));

    await waitFor(() =>
      expect(AutomationService.validateDraft).toHaveBeenCalledWith(
        expect.objectContaining({
          draft: expect.objectContaining({
            repos: [
              { url: "OpenHands/OpenHands", provider: "github" },
              { url: "https://gitlab.com/acme/project" },
            ],
          }),
        }),
      ),
    );
  });

  it("creates plugin drafts with the selected plugin source", async () => {
    vi.mocked(AutomationService.createAutomationDraft).mockResolvedValue({
      id: "automation-1",
    });

    const user = userEvent.setup();
    renderPanel({
      prompt: "Summarize Slack blockers",
      kind: "plugin",
      plugins: ["github:org/blockers-plugin"],
    });

    await user.click(screen.getByTestId("automation-setup-create"));

    await waitFor(() =>
      expect(AutomationService.createAutomationDraft).toHaveBeenCalledWith(
        expect.objectContaining({
          enabled: false,
          prompt: "Summarize Slack blockers",
          plugins: [{ source: "github:org/blockers-plugin" }],
        }),
        "plugin",
      ),
    );
    expect(mockNavigate).toHaveBeenCalledWith("/automations/automation-1");
  });

  it("creates custom bundle drafts with entrypoint and setup script path", async () => {
    vi.mocked(AutomationService.uploadAutomationTarball).mockResolvedValue(
      "oh-internal://uploads/custom-archive",
    );
    vi.mocked(AutomationService.createAutomationDraft).mockResolvedValue({
      id: "automation-custom",
    });

    const user = userEvent.setup();
    renderPanel({
      prompt: "Run a custom security check",
      kind: "custom",
    });

    await user.clear(screen.getByTestId("automation-setup-entrypoint"));
    await user.type(
      screen.getByTestId("automation-setup-entrypoint"),
      "python3 main.py --once",
    );
    await user.click(screen.getByTestId("automation-setup-create"));

    await waitFor(() =>
      expect(AutomationService.uploadAutomationTarball).toHaveBeenCalledWith(
        "Run A Custom Security",
        new Uint8Array([1, 2, 3]),
      ),
    );
    expect(packTarGzip).toHaveBeenCalledWith([
      expect.objectContaining({ name: "main.py", mode: 0o644 }),
      expect.objectContaining({ name: "setup.sh", mode: 0o755 }),
    ]);
    expect(AutomationService.createAutomationDraft).toHaveBeenCalledWith(
      expect.objectContaining({
        tarball_path: "oh-internal://uploads/custom-archive",
        entrypoint: "python3 main.py --once",
        enabled: false,
        setup_script_path: "setup.sh",
      }),
      "custom",
    );
    expect(mockNavigate).toHaveBeenCalledWith("/automations/automation-custom");
  });
});
