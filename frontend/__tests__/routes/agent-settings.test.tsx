import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import userEvent from "@testing-library/user-event";

import AgentSettingsScreen from "#/routes/agent-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import OptionService from "#/api/option-service/option-service.api";
import type { ACPProviderConfig } from "#/api/option-service/option.types";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";

beforeEach(() => {
  useSelectedOrganizationStore.setState({ organizationId: "test-org-id" });
});

afterEach(() => {
  vi.restoreAllMocks();
});

const renderAgentSettings = () =>
  render(
    <MemoryRouter>
      <AgentSettingsScreen />
    </MemoryRouter>,
    {
      wrapper: ({ children }) => (
        <QueryClientProvider client={new QueryClient()}>
          {children}
        </QueryClientProvider>
      ),
    },
  );

// Mirrors the SDK registry-backed web-client config. Frontend tests cannot
// import the Python SDK source directly, but keeping this as a typed fixture
// catches frontend drift in the API shape.
const ACP_PROVIDERS_FIXTURE: ACPProviderConfig[] = [
  {
    key: "claude-code",
    display_name: "Claude Code",
    default_command: ["npx", "-y", "@agentclientprotocol/claude-agent-acp"],
  },
  {
    key: "codex",
    display_name: "Codex",
    default_command: ["npx", "-y", "@zed-industries/codex-acp"],
  },
  {
    key: "gemini-cli",
    display_name: "Gemini CLI",
    default_command: ["npx", "-y", "@google/gemini-cli", "--acp"],
  },
];

const baseConfig = {
  app_mode: "oss" as const,
  posthog_client_key: null,
  feature_flags: {
    enable_billing: false,
    hide_llm_settings: false,
    enable_jira: false,
    enable_jira_dc: false,
    enable_linear: false,
    hide_users_page: false,
    hide_billing_page: false,
    hide_integrations_page: false,
    enable_acp: true,
    enable_onboarding: false,
  },
  providers_configured: [],
  maintenance_start_time: null,
  auth_url: null,
  recaptcha_site_key: null,
  faulty_models: [],
  error_message: null,
  updated_at: "2026-01-01T00:00:00Z",
  github_app_slug: null,
  acp_providers: ACP_PROVIDERS_FIXTURE,
};

describe("AgentSettingsScreen — minimal generic ACP UX", () => {
  it("hydrates the form from saved ACP settings", async () => {
    vi.spyOn(OptionService, "getConfig").mockResolvedValue(baseConfig);
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      agent_settings: {
        agent_kind: "acp",
        acp_server: "custom",
        acp_command: ["npx", "-y", "@agentclientprotocol/claude-agent-acp"],
        acp_args: [],
        acp_env: { ANTHROPIC_API_KEY: "sk-test" },
        acp_model: "claude-opus-4",
      },
    });

    renderAgentSettings();

    await waitFor(() => {
      expect(
        (screen.getByTestId("agent-command-input") as HTMLTextAreaElement)
          .value,
      ).toBe("npx -y @agentclientprotocol/claude-agent-acp");
    });
    expect(
      (screen.getByTestId("agent-model-input") as HTMLInputElement).value,
    ).toBe("claude-opus-4");
  });

  it("clears acp_* fields when switching back to OpenHands", async () => {
    vi.spyOn(OptionService, "getConfig").mockResolvedValue(baseConfig);
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      agent_settings: {
        agent_kind: "acp",
        acp_server: "custom",
        acp_command: ["claude-agent-acp"],
        acp_args: [],
        acp_env: { ANTHROPIC_API_KEY: "sk-test" },
        acp_model: "claude-opus-4",
      },
    });
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderAgentSettings();

    await waitFor(() => {
      expect(
        (screen.getByTestId("agent-command-input") as HTMLTextAreaElement)
          .value,
      ).toBe("claude-agent-acp");
    });

    const dropdown = screen.getByTestId("agent-type-selector");
    await userEvent.click(dropdown);
    const ohOption = await screen.findByRole("option", {
      name: "SETTINGS$AGENT_TYPE_OPENHANDS",
    });
    await userEvent.click(ohOption);

    await userEvent.click(screen.getByTestId("agent-save-button"));

    await waitFor(() => {
      expect(saveSpy).toHaveBeenCalledTimes(1);
    });

    // OH-switch payload sends only ``agent_kind`` — the backend
    // ``Settings.update()`` starts a fresh base when the kind flips and
    // discards any incoming ``acp_*`` fields anyway.
    expect(saveSpy.mock.calls[0][0]).toEqual({
      agent_settings_diff: { agent_kind: "openhands" },
    });
  });

  it("saves built-in presets via acp_server and lets the SDK resolve defaults", async () => {
    vi.spyOn(OptionService, "getConfig").mockResolvedValue(baseConfig);
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      agent_settings: {
        agent_kind: "acp",
        acp_server: "custom",
        acp_command: ["npx", "-y", "@agentclientprotocol/claude-agent-acp"],
        acp_args: [],
        acp_env: {},
        acp_model: null,
      },
    });
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderAgentSettings();

    await waitFor(() => {
      expect(
        (screen.getByTestId("agent-command-input") as HTMLTextAreaElement)
          .value,
      ).toBe("npx -y @agentclientprotocol/claude-agent-acp");
    });

    const commandInput = screen.getByTestId("agent-command-input");
    await userEvent.clear(commandInput);
    await userEvent.type(
      commandInput,
      "npx   -y{enter}@agentclientprotocol/claude-agent-acp",
    );
    await userEvent.type(screen.getByTestId("agent-model-input"), "opus");
    await userEvent.click(screen.getByTestId("agent-save-button"));

    await waitFor(() => {
      expect(saveSpy).toHaveBeenCalledTimes(1);
    });

    expect(saveSpy.mock.calls[0][0]).toMatchObject({
      agent_settings_diff: {
        agent_kind: "acp",
        acp_server: "claude-code",
        acp_command: [],
        acp_model: "opus",
      },
    });
  });

  it("saves as custom when acp_server is built-in but command differs from default", async () => {
    // If the persisted command doesn't match the provider default, the UI must
    // treat it as "custom" and save it that way — even if acp_server says
    // "claude-code". This prevents silent round-trip corruption where the UI
    // shows a built-in preset but saving overwrites the custom command with [].
    vi.spyOn(OptionService, "getConfig").mockResolvedValue(baseConfig);
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      agent_settings: {
        agent_kind: "acp",
        acp_server: "claude-code",
        acp_command: ["npx", "-y", "@custom/my-agent"],
        acp_args: [],
        acp_env: {},
        acp_model: null,
      },
    });
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderAgentSettings();

    await waitFor(() => {
      expect(
        (screen.getByTestId("agent-command-input") as HTMLTextAreaElement)
          .value,
      ).toBe("npx -y @custom/my-agent");
    });

    // Touch the model field to dirty the form (so save is enabled).
    await userEvent.type(screen.getByTestId("agent-model-input"), "x");
    await userEvent.clear(screen.getByTestId("agent-model-input"));
    await userEvent.click(screen.getByTestId("agent-save-button"));

    await waitFor(() => {
      expect(saveSpy).toHaveBeenCalledTimes(1);
    });

    // Since the command doesn't match the claude-code default, it must save as
    // custom with the full command — NOT as acp_server="claude-code" + acp_command=[].
    expect(saveSpy.mock.calls[0][0]).toMatchObject({
      agent_settings_diff: {
        acp_server: "custom",
        acp_command: ["npx", "-y", "@custom/my-agent"],
      },
    });
  });

  it("keeps a useful command placeholder if no provider metadata is available", async () => {
    vi.spyOn(OptionService, "getConfig").mockResolvedValue({
      ...baseConfig,
      acp_providers: [],
    });
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      agent_settings: {
        agent_kind: "acp",
        acp_server: "custom",
        acp_command: [],
        acp_args: [],
        acp_env: {},
        acp_model: null,
      },
    });

    renderAgentSettings();

    expect(await screen.findByTestId("agent-command-input")).toHaveAttribute(
      "placeholder",
      "npx -y <package-name>",
    );
  });
});
