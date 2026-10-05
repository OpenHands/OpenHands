import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import ToolCatalogService from "#/api/tool-catalog-service/tool-catalog-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import {
  AgentSettingsScreen,
  type AgentSettingsSaveControl,
} from "#/routes/agent-settings";

const CATALOG = [
  {
    name: "terminal",
    user_selectable: true,
    usable: true,
    in_default_set: true,
    description: "Run shell commands",
  },
];

beforeEach(() => {
  vi.restoreAllMocks();
  vi.spyOn(ToolCatalogService, "getCatalog").mockResolvedValue(CATALOG);
});

afterEach(() => {
  setActiveSelection(null);
  setRegisteredBackends([]);
  __resetActiveStoreForTests();
});

it.each([false, true])(
  "initializes a custom selection once the catalog arrives (explicitly cleared: %s)",
  async (clearWhilePending) => {
    const user = userEvent.setup();
    let resolveCatalog!: (catalog: typeof CATALOG) => void;
    vi.mocked(ToolCatalogService.getCatalog).mockReturnValue(
      new Promise((resolve) => {
        resolveCatalog = resolve;
      }),
    );
    let control: AgentSettingsSaveControl | null = null;
    render(
      <MemoryRouter>
        <QueryClientProvider
          client={
            new QueryClient({ defaultOptions: { queries: { retry: false } } })
          }
        >
          <AgentSettingsScreen
            agentSettingsOverride={{
              agent_kind: "openhands",
              tools: clearWhilePending ? [] : null,
            }}
            onSaveControlChange={(next) => {
              control = next;
            }}
          />
        </QueryClientProvider>
      </MemoryRouter>,
    );
    if (!clearWhilePending) {
      expect(
        await screen.findByTestId("agent-settings-tools-mode"),
      ).toBeDisabled();
    }
    await act(async () => resolveCatalog(CATALOG));
    if (!clearWhilePending) {
      await user.click(screen.getByTestId("agent-settings-tools-mode"));
      await user.click(
        await screen.findByRole("option", {
          name: "SETTINGS$AGENT_PROFILE_TOOLS_CHOOSE",
        }),
      );
    }
    await waitFor(() =>
      expect(control!.buildAgentProfileFields()).toMatchObject({
        tools: clearWhilePending ? [] : [{ name: "terminal", params: {} }],
      }),
    );
    expect(control!.isValid).toBe(true);
  },
);

it.each(["saved", "cleared"] as const)(
  "preserves a %s empty tool selection across mode changes",
  async (selection) => {
    const user = userEvent.setup();
    let control: AgentSettingsSaveControl | null = null;
    render(
      <MemoryRouter>
        <QueryClientProvider
          client={
            new QueryClient({
              defaultOptions: { queries: { retry: false } },
            })
          }
        >
          <AgentSettingsScreen
            agentSettingsOverride={{
              agent_kind: "openhands",
              tools: selection === "saved" ? [] : null,
            }}
            onSaveControlChange={(next) => {
              control = next;
            }}
          />
        </QueryClientProvider>
      </MemoryRouter>,
    );
    const setMode = async (mode: "STANDARD" | "CHOOSE") => {
      await user.click(screen.getByTestId("agent-settings-tools-mode"));
      await user.click(
        await screen.findByRole("option", {
          name: `SETTINGS$AGENT_PROFILE_TOOLS_${mode}`,
        }),
      );
    };
    await waitFor(() =>
      expect(
        screen.getByTestId("agent-settings-tools-mode"),
      ).not.toBeDisabled(),
    );
    if (selection === "cleared") {
      await setMode("CHOOSE");
      await user.click(
        await screen.findByTestId("agent-settings-tool-terminal"),
      );
    }

    await setMode("STANDARD");
    await setMode("CHOOSE");

    expect(
      screen.getByTestId("agent-settings-tool-terminal"),
    ).not.toBeChecked();
    expect(control!.buildAgentProfileFields()).toMatchObject({ tools: [] });
  },
);

it("reports a failed catalog load and recovers on retry", async () => {
  const user = userEvent.setup();
  vi.mocked(ToolCatalogService.getCatalog)
    .mockRejectedValueOnce(new Error("proxy 502"))
    .mockResolvedValue(CATALOG);
  render(
    <MemoryRouter>
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <AgentSettingsScreen
          agentSettingsOverride={{ agent_kind: "openhands", tools: null }}
          onSaveControlChange={() => {}}
        />
      </QueryClientProvider>
    </MemoryRouter>,
  );

  expect(
    await screen.findByTestId("agent-settings-tools-load-failed"),
  ).toHaveTextContent("SETTINGS$AGENT_PROFILE_TOOLS_LOAD_FAILED");

  await user.click(screen.getByTestId("agent-settings-tools-retry"));

  expect(
    await screen.findByTestId("agent-settings-tool-terminal"),
  ).toBeChecked();
  expect(
    screen.queryByTestId("agent-settings-tools-load-failed"),
  ).not.toBeInTheDocument();
  expect(screen.getByTestId("agent-settings-tools-mode")).not.toBeDisabled();
});

it("locks a custom selection until the catalog loads", async () => {
  vi.mocked(ToolCatalogService.getCatalog).mockReturnValue(
    new Promise(() => {}),
  );
  render(
    <MemoryRouter>
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <AgentSettingsScreen
          agentSettingsOverride={{
            agent_kind: "openhands",
            tools: [{ name: "terminal", params: {} }],
          }}
          onSaveControlChange={() => {}}
        />
      </QueryClientProvider>
    </MemoryRouter>,
  );

  expect(
    await screen.findByTestId("agent-settings-tool-terminal"),
  ).toBeDisabled();
});

it("describes a stored tool the picker would otherwise filter out", async () => {
  vi.mocked(ToolCatalogService.getCatalog).mockResolvedValue([
    ...CATALOG,
    {
      name: "browser_tool_set",
      user_selectable: true,
      usable: false,
      in_default_set: true,
      description: "Browse the web",
    },
  ]);
  render(
    <MemoryRouter>
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <AgentSettingsScreen
          agentSettingsOverride={{
            agent_kind: "openhands",
            tools: [{ name: "browser_tool_set", params: {} }],
          }}
          onSaveControlChange={() => {}}
        />
      </QueryClientProvider>
    </MemoryRouter>,
  );

  expect(
    await screen.findByTestId("agent-settings-tool-list"),
  ).toHaveTextContent("Browse the web");
});

it("hides tools on a cloud backend and leaves the stored selection alone", async () => {
  setRegisteredBackends([
    {
      id: "cloud",
      name: "cloud",
      host: "https://app.example.test",
      apiKey: "cloud-key",
      kind: "cloud",
    },
  ]);
  setActiveSelection({ backendId: "cloud" });
  let control: AgentSettingsSaveControl | null = null;
  render(
    <MemoryRouter>
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <AgentSettingsScreen
          agentSettingsOverride={{
            agent_kind: "openhands",
            tools: [{ name: "terminal", params: {} }],
          }}
          onSaveControlChange={(next) => {
            control = next;
          }}
        />
      </QueryClientProvider>
    </MemoryRouter>,
  );

  await waitFor(() => expect(control).not.toBeNull());
  expect(
    screen.queryByTestId("agent-settings-tools-mode"),
  ).not.toBeInTheDocument();
  expect(ToolCatalogService.getCatalog).not.toHaveBeenCalled();
  expect(control!.buildAgentProfileFields()).not.toHaveProperty("tools");
});

it("points the default profile to a named profile instead of offering tools", async () => {
  let control: AgentSettingsSaveControl | null = null;
  render(
    <MemoryRouter>
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <AgentSettingsScreen
          agentSettingsOverride={{
            agent_kind: "openhands",
            tools: [{ name: "terminal", params: {} }],
          }}
          onSaveControlChange={(next) => {
            control = next;
          }}
          isDefaultProfile
        />
      </QueryClientProvider>
    </MemoryRouter>,
  );

  expect(
    await screen.findByTestId("agent-settings-tools-default-profile"),
  ).toHaveTextContent("SETTINGS$AGENT_PROFILE_TOOLS_DEFAULT_PROFILE");
  expect(
    screen.queryByTestId("agent-settings-tools-mode"),
  ).not.toBeInTheDocument();
  expect(ToolCatalogService.getCatalog).not.toHaveBeenCalled();
  expect(control!.buildAgentProfileFields()).not.toHaveProperty("tools");
});
