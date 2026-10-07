import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { WorkspaceFileDiscoverySettings } from "#/components/features/files-tab/workspace-file-discovery-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { DEFAULT_FILE_DISCOVERY } from "#/utils/workspace-file-discovery";
import { useSettings } from "#/hooks/query/use-settings";

vi.mock("#/api/settings-service/settings-service.api", () => ({
  default: { getSettings: vi.fn(), saveSettings: vi.fn() },
}));

const snapshots = {
  local: { active: { backend: { id: "local", kind: "local" }, orgId: null } },
  cloud: { active: { backend: { id: "cloud", kind: "cloud" }, orgId: null } },
};
let backend: keyof typeof snapshots = "local";
vi.mock("#/api/backend-registry/active-store", () => ({
  subscribeActiveBackend: () => () => {},
  getSnapshot: () => snapshots[backend],
  isNoBackend: () => false,
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => snapshots[backend].active,
}));

function SettingsConsumer() {
  const { data } = useSettings();
  return (
    <output aria-label="MCP settings">
      {JSON.stringify(data?.mcp_config)}
    </output>
  );
}

function setup() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const view = (path: string) => (
    <QueryClientProvider client={client}>
      <WorkspaceFileDiscoverySettings workingDir={path} />
    </QueryClientProvider>
  );
  const rendered = render(view("/project"));
  return {
    user: userEvent.setup(),
    navigate: (path: string) => rendered.rerender(view(path)),
  };
}

// @spec WFD-002 — Workspace-scoped server persistence
describe("workspace discovery settings", () => {
  beforeEach(() => {
    backend = "local";
    vi.mocked(SettingsService.getSettings)
      .mockReset()
      .mockResolvedValue(DEFAULT_SETTINGS);
    vi.mocked(SettingsService.saveSettings).mockReset().mockResolvedValue(true);
  });

  it("saves edited options for only the current workspace and reports failed saves", async () => {
    const { user } = setup();
    await user.click(
      await screen.findByRole("button", { name: "FILES$DISCOVERY_SETTINGS" }),
    );
    await user.clear(screen.getByRole("spinbutton"));
    await user.type(screen.getByRole("spinbutton"), "0");
    await user.clear(screen.getByRole("textbox"));
    await user.type(screen.getByRole("textbox"), "*/bin\nobj");
    await user.click(screen.getByRole("checkbox"));
    vi.mocked(SettingsService.saveSettings).mockRejectedValueOnce(
      new Error("offline"),
    );
    await user.click(screen.getByRole("button", { name: "BUTTON$SAVE" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "FILES$DISCOVERY_SAVE_ERROR",
    );
    await user.click(screen.getByRole("button", { name: "BUTTON$SAVE" }));
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    );
    expect(SettingsService.saveSettings).toHaveBeenLastCalledWith({
      workspace_file_discovery: {
        "/project": {
          maxFiles: 0,
          includeSymlinks: true,
          excludedPatterns: ["*/bin", "obj"],
        },
      },
    });
  });

  it("hydrates saved options, discards canceled drafts and restores defaults only on Save", async () => {
    vi.mocked(SettingsService.getSettings).mockResolvedValue({
      ...DEFAULT_SETTINGS,
      workspace_file_discovery: {
        "/project": {
          excludedPatterns: [],
          maxFiles: 3,
          includeSymlinks: true,
        },
      },
    });
    const { user } = setup();
    await user.click(
      await screen.findByRole("button", { name: "FILES$DISCOVERY_SETTINGS" }),
    );
    expect(screen.getByRole("spinbutton")).toHaveValue(3);
    await user.click(
      screen.getByRole("button", { name: "FILES$DISCOVERY_RESET" }),
    );
    await user.click(screen.getByRole("button", { name: "BUTTON$CANCEL" }));
    expect(SettingsService.saveSettings).not.toHaveBeenCalled();
    await user.click(
      screen.getByRole("button", { name: "FILES$DISCOVERY_SETTINGS" }),
    );
    expect(screen.getByRole("spinbutton")).toHaveValue(3);
    await user.click(
      screen.getByRole("button", { name: "FILES$DISCOVERY_RESET" }),
    );
    await user.click(screen.getByRole("button", { name: "BUTTON$SAVE" }));
    await waitFor(() =>
      expect(SettingsService.saveSettings).toHaveBeenCalledWith({
        workspace_file_discovery: { "/project": DEFAULT_FILE_DISCOVERY },
      }),
    );
  });

  it("discards an editor on workspace navigation and hides it on Cloud", async () => {
    const { user, navigate } = setup();
    await user.click(
      await screen.findByRole("button", { name: "FILES$DISCOVERY_SETTINGS" }),
    );
    navigate("/other");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    backend = "cloud";
    navigate("/other");
    expect(
      screen.queryByRole("button", { name: "FILES$DISCOVERY_SETTINGS" }),
    ).not.toBeInTheDocument();
    expect(SettingsService.saveSettings).not.toHaveBeenCalled();
  });

  it("preserves normalized settings when file discovery mounts and saves", async () => {
    const mcpConfig = {
      docs: { transport: "sse" as const, url: "https://docs.example.test/sse" },
    };
    vi.mocked(SettingsService.getSettings).mockResolvedValue({
      ...DEFAULT_SETTINGS,
      agent_settings: { mcp_config: mcpConfig },
    });
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const view = (showFiles: boolean) => (
      <QueryClientProvider client={client}>
        <SettingsConsumer />
        {showFiles && <WorkspaceFileDiscoverySettings workingDir="/project" />}
      </QueryClientProvider>
    );
    const rendered = render(view(false));
    await waitFor(() =>
      expect(
        screen.getByRole("status", { name: "MCP settings" }),
      ).toHaveTextContent(JSON.stringify(mcpConfig)),
    );

    rendered.rerender(view(true));
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "FILES$DISCOVERY_SETTINGS" }),
    );
    await user.click(screen.getByRole("button", { name: "BUTTON$SAVE" }));
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    );

    expect(SettingsService.getSettings).toHaveBeenCalledTimes(2);
    expect(
      screen.getByRole("status", { name: "MCP settings" }),
    ).toHaveTextContent(JSON.stringify(mcpConfig));
  });
});
