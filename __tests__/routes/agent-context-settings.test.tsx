import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import AgentContextSettingsScreen from "#/routes/agent-context-settings";
import { Settings } from "#/types/settings";

function buildSettings(overrides: Partial<Settings> = {}): Settings {
  return {
    ...MOCK_DEFAULT_USER_SETTINGS,
    ...overrides,
    agent_settings: {
      ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
      ...overrides.agent_settings,
    },
    agent_settings_schema:
      overrides.agent_settings_schema ??
      MOCK_DEFAULT_USER_SETTINGS.agent_settings_schema,
  };
}

function renderAgentContextSettingsScreen() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return render(<AgentContextSettingsScreen />, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
}

beforeEach(() => {
  vi.restoreAllMocks();
});

describe("AgentContextSettingsScreen", () => {
  it("renders the persistent-memory toggle from the agent schema, off by default", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());

    renderAgentContextSettingsScreen();

    await screen.findByTestId("agent-context-settings-screen");

    const toggle = screen.getByTestId("sdk-settings-agent_context.load_memory");
    expect(toggle).toBeInTheDocument();
    expect(toggle).not.toBeChecked();
  });

  it("reflects a stored agent_context.load_memory value", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          agent_context: { load_memory: true },
        },
      }),
    );

    renderAgentContextSettingsScreen();

    await screen.findByTestId("agent-context-settings-screen");

    expect(
      screen.getByTestId("sdk-settings-agent_context.load_memory"),
    ).toBeChecked();
  });

  it("saves the toggle as a nested agent_settings_diff", async () => {
    const user = userEvent.setup();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderAgentContextSettingsScreen();

    await screen.findByTestId("agent-context-settings-screen");
    await user.click(
      screen.getByTestId("sdk-settings-agent_context.load_memory"),
    );
    await user.click(screen.getByTestId("save-button"));

    expect(saveSettingsSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        agent_settings_diff: { agent_context: { load_memory: true } },
      }),
    );
  });

  it("shows the toggle without a dead Basic tab (major-only section)", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());

    renderAgentContextSettingsScreen();

    await screen.findByTestId("agent-context-settings-screen");

    // The section's only field is major-prominence: the page floors its
    // initial view at "advanced" and hides the view toggle entirely (a
    // Basic tab would render an empty page).
    expect(
      screen.getByTestId("sdk-settings-agent_context.load_memory"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("sdk-section-basic-toggle"),
    ).not.toBeInTheDocument();
  });

  it("shows the schema-unavailable state when the server does not expose the section", async () => {
    const schemaWithoutAgentContext = structuredClone(
      MOCK_DEFAULT_USER_SETTINGS.agent_settings_schema!,
    );
    schemaWithoutAgentContext.sections =
      schemaWithoutAgentContext.sections.filter(
        (section) => section.key !== "agent_context",
      );
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ agent_settings_schema: schemaWithoutAgentContext }),
    );

    renderAgentContextSettingsScreen();

    expect(await screen.findByTestId("sdk-schema-unavailable")).toBeVisible();
    expect(
      screen.queryByTestId("sdk-settings-agent_context.load_memory"),
    ).not.toBeInTheDocument();
  });
});
