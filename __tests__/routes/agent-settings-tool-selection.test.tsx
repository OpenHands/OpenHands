import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { beforeEach, expect, it, vi } from "vitest";
import ToolCatalogService from "#/api/tool-catalog-service/tool-catalog-service.api";
import * as profileSupport from "#/api/agent-profiles-service/profile-field-support";
import {
  AgentSettingsScreen,
  type AgentSettingsSaveControl,
} from "#/routes/agent-settings";

beforeEach(() => {
  vi.restoreAllMocks();
  vi.spyOn(profileSupport, "agentProfileSupportsToolCatalog").mockReturnValue(
    true,
  );
  vi.spyOn(ToolCatalogService, "getCatalog").mockResolvedValue([
    { name: "terminal", user_selectable: true, usable: true },
  ]);
  vi.spyOn(ToolCatalogService, "getResolvedToolNames").mockResolvedValue([
    "terminal",
  ]);
});

it.each([false, true])(
  "initializes an unnamed custom profile after resolution (explicitly cleared: %s)",
  async (clearWhilePending) => {
    const user = userEvent.setup();
    let resolveNames!: (names: string[]) => void;
    vi.mocked(ToolCatalogService.getResolvedToolNames).mockReturnValue(
      new Promise<string[]>((resolve) => {
        resolveNames = resolve;
      }),
    );
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const settings = { agent_kind: "openhands", tools: null };
    let control: AgentSettingsSaveControl | null = null;
    const editor = (name: string) => (
      <MemoryRouter>
        <QueryClientProvider client={client}>
          <AgentSettingsScreen
            embedded
            profileName={name}
            llmProfileRef="main"
            agentSettingsOverride={settings}
            onSaveControlChange={(next) => {
              control = next;
            }}
          />
        </QueryClientProvider>
      </MemoryRouter>
    );
    const { rerender } = render(editor(""));
    await user.click(await screen.findByTestId("agent-settings-tools-mode"));
    await user.click(
      await screen.findByRole("option", {
        name: "SETTINGS$AGENT_PROFILE_TOOLS_CHOOSE",
      }),
    );
    const terminal = await screen.findByTestId("agent-settings-tool-terminal");
    expect(ToolCatalogService.getResolvedToolNames).not.toHaveBeenCalled();
    if (clearWhilePending) {
      await user.click(terminal);
      await user.click(terminal);
    }
    rerender(editor("new-agent"));
    await waitFor(() =>
      expect(ToolCatalogService.getResolvedToolNames).toHaveBeenCalled(),
    );
    expect(control!.isValid).toBe(clearWhilePending);
    await act(async () => resolveNames(["terminal"]));
    await waitFor(() =>
      expect(control!.buildAgentProfileFields()).toMatchObject({
        tools: clearWhilePending ? [] : [{ name: "terminal", params: {} }],
      }),
    );
    expect(terminal).toHaveProperty("checked", !clearWhilePending);
    expect(control!.isValid).toBe(true);
  },
);

it("waits for a usable standard set and the catalog before initializing", async () => {
  const user = userEvent.setup();
  let resolveCatalog!: (
    catalog: Awaited<ReturnType<typeof ToolCatalogService.getCatalog>>,
  ) => void;
  vi.mocked(ToolCatalogService.getCatalog).mockReturnValue(
    new Promise((resolve) => {
      resolveCatalog = resolve;
    }),
  );
  vi.mocked(ToolCatalogService.getResolvedToolNames).mockResolvedValue([]);
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const settings = { agent_kind: "openhands", tools: null };
  let control: AgentSettingsSaveControl | null = null;
  const editor = (name: string) => (
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <AgentSettingsScreen
          embedded
          profileName={name}
          llmProfileRef="main"
          agentSettingsOverride={settings}
          onSaveControlChange={(next) => {
            control = next;
          }}
        />
      </QueryClientProvider>
    </MemoryRouter>
  );
  const { rerender } = render(editor(""));
  await user.click(await screen.findByTestId("agent-settings-tools-mode"));
  await user.click(
    await screen.findByRole("option", {
      name: "SETTINGS$AGENT_PROFILE_TOOLS_CHOOSE",
    }),
  );
  rerender(editor("empty-response"));
  await waitFor(() =>
    expect(ToolCatalogService.getResolvedToolNames).toHaveBeenCalled(),
  );
  expect(control!.isValid).toBe(false);

  vi.mocked(ToolCatalogService.getResolvedToolNames).mockResolvedValue([
    "terminal",
  ]);
  rerender(editor("resolved-response"));
  await waitFor(() =>
    expect(ToolCatalogService.getResolvedToolNames).toHaveBeenCalledTimes(2),
  );
  expect(control!.isValid).toBe(false);
  await act(async () =>
    resolveCatalog([{ name: "terminal", user_selectable: true, usable: true }]),
  );
  await waitFor(() =>
    expect(control!.buildAgentProfileFields()).toMatchObject({
      tools: [{ name: "terminal", params: {} }],
    }),
  );
  expect(control!.isValid).toBe(true);
});

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
            embedded
            profileName="bare-agent"
            llmProfileRef="main"
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
