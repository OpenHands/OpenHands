import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import RuntimeSettingsScreen from "#/routes/runtime-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { Settings } from "#/types/settings";

const workspaceState = vi.hoisted(() => ({
  isolated: false,
  dockerSelectable: true,
  runtimeSettingsApplied: true,
}));

vi.mock("#/hooks/query/use-conversation-workspace", () => ({
  useConversationWorkspace: () => ({
    ...workspaceState,
    unsupportedMessage: null,
  }),
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({ backend: { kind: "local" }, orgId: null }),
}));

function settings(overrides: Partial<Settings> = {}): Settings {
  return { ...MOCK_DEFAULT_USER_SETTINGS, ...overrides };
}

function renderScreen() {
  return render(<RuntimeSettingsScreen />, {
    wrapper: ({ children }) => (
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        {children}
      </QueryClientProvider>
    ),
  });
}

describe("RuntimeSettingsScreen", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    Object.assign(workspaceState, {
      isolated: false,
      dockerSelectable: true,
      runtimeSettingsApplied: true,
    });
  });

  it("saves the default mode with docker and worktree settings", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(settings());
    const save = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);
    renderScreen();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("combobox", {
        name: "SETTINGS$DEFAULT_WORKSPACE_MODE",
      }),
    );
    await user.click(
      await screen.findByText("COMMON$WORKSPACE_MODE_DOCKER_CONTAINER"),
    );
    await user.type(screen.getByTestId("docker-retention-days-input"), "14");
    await user.type(screen.getByTestId("docker-memory-input"), "8g");
    await user.type(screen.getByTestId("worktree-retention-days-input"), "30");
    await user.click(screen.getByTestId("worktree-remove-on-delete-switch"));
    await user.click(screen.getByTestId("submit-button"));

    await waitFor(() =>
      expect(save).toHaveBeenCalledWith({
        default_workspace_mode: "docker_container",
        runtime_settings: {
          docker_retention_days: 14,
          docker_memory: "8g",
          worktree_retention_days: 30,
          worktree_remove_on_delete: true,
        },
      }),
    );
  });

  it("disables docker options and hides the docker mode when unavailable", async () => {
    workspaceState.dockerSelectable = false;
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(settings());
    renderScreen();

    expect(
      await screen.findByText("SETTINGS$DOCKER_RUNTIME_UNAVAILABLE"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("docker-memory-input")).toBeDisabled();
    expect(screen.getByTestId("worktree-retention-days-input")).toBeEnabled();
    await userEvent
      .setup()
      .click(
        screen.getByRole("combobox", {
          name: "SETTINGS$DEFAULT_WORKSPACE_MODE",
        }),
      );
    expect(
      screen.queryByText("COMMON$WORKSPACE_MODE_DOCKER_CONTAINER"),
    ).toBeNull();
  });

  it("warns when the agent-server does not apply runtime settings", async () => {
    workspaceState.runtimeSettingsApplied = false;
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      settings({ runtime_settings: { worktree_base: "head" } }),
    );
    renderScreen();

    expect(
      await screen.findByTestId("runtime-settings-not-applied"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("submit-button")).toBeDisabled();
  });
});
