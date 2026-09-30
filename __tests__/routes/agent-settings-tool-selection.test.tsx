import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { beforeEach, expect, it, vi } from "vitest";
import ToolCatalogService from "#/api/tool-catalog-service/tool-catalog-service.api";
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
  },
];

beforeEach(() => {
  vi.restoreAllMocks();
  vi.spyOn(ToolCatalogService, "getCatalog").mockResolvedValue(CATALOG);
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
