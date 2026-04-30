import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AgentSettingsScreen, {
  buildAgentSettingsDiff,
} from "#/routes/agent-settings";

const mockSettings = vi.hoisted(() => ({
  data: null as Record<string, unknown> | null,
  isLoading: false,
}));

const mockSaveSettings = vi.hoisted(() => vi.fn());
const mockIsPending = vi.hoisted(() => ({ value: false }));
const mockConfig = vi.hoisted(() => ({
  data: {
    feature_flags: { enable_acp: true },
  },
  isLoading: false,
}));
const mockNavigate = vi.hoisted(() => vi.fn());

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => mockSettings,
}));

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => mockConfig,
}));

vi.mock("#/hooks/mutation/use-save-settings", () => ({
  useSaveSettings: () => ({
    mutate: mockSaveSettings,
    isPending: mockIsPending.value,
  }),
}));

vi.mock("react-router", async () => {
  const actual =
    await vi.importActual<typeof import("react-router")>("react-router");
  return {
    ...actual,
    useNavigate: () => mockNavigate,
    useRevalidator: () => ({ revalidate: vi.fn() }),
  };
});

const queryClient = new QueryClient();
const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
);

function renderScreen() {
  return render(<AgentSettingsScreen />, { wrapper });
}

describe("AgentSettingsScreen", () => {
  beforeEach(() => {
    queryClient.clear();
    vi.clearAllMocks();
    mockSettings.data = null;
    mockSettings.isLoading = false;
    mockConfig.data = { feature_flags: { enable_acp: true } };
    mockConfig.isLoading = false;
    mockIsPending.value = false;
  });

  describe("initial state", () => {
    it("renders the page title", () => {
      renderScreen();
      expect(screen.getByText("SETTINGS$AGENT_PAGE_TITLE")).toBeInTheDocument();
    });

    it("shows Basic tab by default", () => {
      renderScreen();
      expect(screen.getByText("SETTINGS$AGENT_BASIC_TAB")).toBeInTheDocument();
    });

    it("does not show Advanced tab when OpenHands is selected", () => {
      renderScreen();
      expect(
        screen.queryByText("SETTINGS$AGENT_ADVANCED_TAB"),
      ).not.toBeInTheDocument();
    });

    it("shows agent type dropdown", () => {
      renderScreen();
      expect(screen.getByTestId("agent-type-selector")).toBeInTheDocument();
    });

    it("does not show API key field when OpenHands is selected", () => {
      renderScreen();
      expect(
        screen.queryByTestId("agent-api-key-input"),
      ).not.toBeInTheDocument();
    });

    it("returns null while loading", () => {
      mockSettings.isLoading = true;
      const { container } = renderScreen();
      expect(container.firstChild).toBeNull();
    });

    it("returns null when ACP feature flag is disabled", () => {
      mockConfig.data = { feature_flags: { enable_acp: false } };
      const { container } = renderScreen();
      expect(container.firstChild).toBeNull();
      expect(mockNavigate).toHaveBeenCalledWith("/settings", {
        replace: true,
      });
    });
  });

  describe("loading from saved ACP settings", () => {
    it("restores claude-code agent type from saved settings", () => {
      mockSettings.data = {
        agent_settings: { agent_kind: "acp", acp_server: "claude-code" },
      } as Record<string, unknown>;
      renderScreen();
      // API key field should be visible for claude-code
      expect(screen.getByTestId("agent-api-key-input")).toBeInTheDocument();
      expect(screen.getByText("Anthropic API Key")).toBeInTheDocument();
    });

    it("falls back to claude-code for unknown acp_server", () => {
      mockSettings.data = {
        agent_settings: { agent_kind: "acp", acp_server: "unknown-server" },
      } as Record<string, unknown>;
      renderScreen();
      // Should fall back to claude-code, showing Anthropic API key
      expect(screen.getByText("Anthropic API Key")).toBeInTheDocument();
    });

    it("shows openhands when saved kind is llm", () => {
      mockSettings.data = {
        agent_settings: { agent_kind: "llm" },
      } as Record<string, unknown>;
      renderScreen();
      expect(
        screen.queryByTestId("agent-api-key-input"),
      ).not.toBeInTheDocument();
    });
  });

  describe("API key field visibility per provider", () => {
    it("shows Anthropic API key for claude-code", async () => {
      mockSettings.data = {
        agent_settings: { agent_kind: "acp", acp_server: "claude-code" },
      } as Record<string, unknown>;
      renderScreen();
      expect(screen.getByText("Anthropic API Key")).toBeInTheDocument();
    });

    it("shows OpenAI API key for codex", () => {
      mockSettings.data = {
        agent_settings: { agent_kind: "acp", acp_server: "codex" },
      } as Record<string, unknown>;
      renderScreen();
      expect(screen.getByText("OpenAI API Key")).toBeInTheDocument();
    });

    it("shows Google API key for gemini-cli", () => {
      mockSettings.data = {
        agent_settings: { agent_kind: "acp", acp_server: "gemini-cli" },
      } as Record<string, unknown>;
      renderScreen();
      expect(screen.getByText("Google API Key")).toBeInTheDocument();
    });
  });

  describe("base URL field", () => {
    it("shows Anthropic Base URL for claude-code", () => {
      mockSettings.data = {
        agent_settings: { agent_kind: "acp", acp_server: "claude-code" },
      } as Record<string, unknown>;
      renderScreen();
      expect(screen.getByText("Anthropic Base URL")).toBeInTheDocument();
      expect(screen.getByTestId("agent-base-url-input")).toBeInTheDocument();
    });

    it("shows OpenAI Base URL for codex", () => {
      mockSettings.data = {
        agent_settings: { agent_kind: "acp", acp_server: "codex" },
      } as Record<string, unknown>;
      renderScreen();
      expect(screen.getByText("OpenAI Base URL")).toBeInTheDocument();
    });

    it("shows Google Base URL for gemini-cli", () => {
      mockSettings.data = {
        agent_settings: { agent_kind: "acp", acp_server: "gemini-cli" },
      } as Record<string, unknown>;
      renderScreen();
      expect(screen.getByText("Google Base URL")).toBeInTheDocument();
    });

    it("does not show base URL field for OpenHands", () => {
      renderScreen();
      expect(
        screen.queryByTestId("agent-base-url-input"),
      ).not.toBeInTheDocument();
    });

    it("populates base URL from saved settings", async () => {
      mockSettings.data = {
        agent_settings: {
          agent_kind: "acp",
          acp_server: "claude-code",
          llm: { base_url: "https://my-proxy.example.com" },
        },
      } as Record<string, unknown>;
      renderScreen();
      await waitFor(() => {
        const input = screen.getByTestId(
          "agent-base-url-input",
        ) as HTMLInputElement;
        expect(input.value).toBe("https://my-proxy.example.com");
      });
    });

    it("includes base_url in save payload when entered", async () => {
      const user = userEvent.setup();
      mockSettings.data = {
        agent_settings: { agent_kind: "acp", acp_server: "claude-code" },
      } as Record<string, unknown>;
      renderScreen();
      const urlInput = screen.getByTestId("agent-base-url-input");
      await user.type(urlInput, "https://proxy.example.com");
      const saveBtn = screen.getByTestId("agent-save-button");
      await waitFor(() => expect(saveBtn).not.toBeDisabled());
      await user.click(saveBtn);
      expect(mockSaveSettings).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_settings_diff: expect.objectContaining({
            llm: expect.objectContaining({
              base_url: "https://proxy.example.com",
            }),
          }),
        }),
        expect.any(Object),
      );
    });

    it("omits llm.base_url from payload when field is empty", async () => {
      const user = userEvent.setup();
      mockSettings.data = {
        agent_settings: { agent_kind: "acp", acp_server: "claude-code" },
      } as Record<string, unknown>;
      renderScreen();
      const apiKeyInput = screen.getByTestId("agent-api-key-input");
      await user.type(apiKeyInput, "sk-ant-test");
      const saveBtn = screen.getByTestId("agent-save-button");
      await waitFor(() => expect(saveBtn).not.toBeDisabled());
      await user.click(saveBtn);
      const diff = mockSaveSettings.mock.calls[0][0].agent_settings_diff;
      expect(diff.llm).not.toHaveProperty("base_url");
    });
  });

  describe("Advanced tab", () => {
    beforeEach(() => {
      mockSettings.data = {
        agent_settings: { agent_kind: "acp", acp_server: "claude-code" },
      } as Record<string, unknown>;
    });

    it("shows Advanced tab button when ACP agent is selected", () => {
      renderScreen();
      expect(
        screen.getByText("SETTINGS$AGENT_ADVANCED_TAB"),
      ).toBeInTheDocument();
    });

    it("shows command/args/env fields after clicking Advanced tab", async () => {
      const user = userEvent.setup();
      renderScreen();
      await user.click(screen.getByText("SETTINGS$AGENT_ADVANCED_TAB"));
      expect(screen.getByTestId("agent-command-input")).toBeInTheDocument();
      expect(screen.getByTestId("agent-args-input")).toBeInTheDocument();
      expect(screen.getByTestId("agent-env-input")).toBeInTheDocument();
    });
  });

  describe("env JSON validation", () => {
    beforeEach(async () => {
      mockSettings.data = {
        agent_settings: { agent_kind: "acp", acp_server: "claude-code" },
      } as Record<string, unknown>;
    });

    it("shows error for invalid JSON", async () => {
      const user = userEvent.setup();
      renderScreen();
      await user.click(screen.getByText("SETTINGS$AGENT_ADVANCED_TAB"));
      const textarea = screen.getByTestId("agent-env-input");
      await user.clear(textarea);
      await user.type(textarea, "not json");
      await waitFor(() => {
        expect(
          screen.getByText("SETTINGS$MCP_ERROR_INVALID_JSON"),
        ).toBeInTheDocument();
      });
    });

    it("shows error when JSON is an array not an object", async () => {
      const user = userEvent.setup();
      renderScreen();
      await user.click(screen.getByText("SETTINGS$AGENT_ADVANCED_TAB"));
      const textarea = screen.getByTestId("agent-env-input");
      // Use fireEvent.change to set the full value at once, avoiding
      // user-event's bracket escaping for characters like { and [
      fireEvent.change(textarea, { target: { value: "[1,2,3]" } });
      await waitFor(() => {
        expect(
          screen.getByText("SETTINGS$AGENT_ENV_MUST_BE_OBJECT"),
        ).toBeInTheDocument();
      });
    });

    it("shows error when JSON values are not all strings", async () => {
      const user = userEvent.setup();
      renderScreen();
      await user.click(screen.getByText("SETTINGS$AGENT_ADVANCED_TAB"));
      const textarea = screen.getByTestId("agent-env-input");
      fireEvent.change(textarea, { target: { value: '{"KEY":123}' } });
      await waitFor(() => {
        expect(
          screen.getByText("SETTINGS$AGENT_ENV_VALUES_MUST_BE_STRINGS"),
        ).toBeInTheDocument();
      });
    });

    it("clears error when valid JSON object is entered", async () => {
      const user = userEvent.setup();
      renderScreen();
      await user.click(screen.getByText("SETTINGS$AGENT_ADVANCED_TAB"));
      const textarea = screen.getByTestId("agent-env-input");
      fireEvent.change(textarea, { target: { value: "not json" } });
      await waitFor(() => {
        expect(
          screen.getByText("SETTINGS$MCP_ERROR_INVALID_JSON"),
        ).toBeInTheDocument();
      });
      fireEvent.change(textarea, {
        target: { value: '{"MY_VAR":"value"}' },
      });
      await waitFor(() => {
        expect(
          screen.queryByText("SETTINGS$MCP_ERROR_INVALID_JSON"),
        ).not.toBeInTheDocument();
        expect(
          screen.queryByText("SETTINGS$AGENT_ENV_MUST_BE_OBJECT"),
        ).not.toBeInTheDocument();
      });
    });
  });

  describe("save button state", () => {
    it("save button is disabled when form is not dirty", () => {
      renderScreen();
      const saveBtn = screen.getByTestId("agent-save-button");
      expect(saveBtn).toBeDisabled();
    });

    it("save button is disabled when there is an env error", async () => {
      const user = userEvent.setup();
      mockSettings.data = {
        agent_settings: { agent_kind: "acp", acp_server: "claude-code" },
      } as Record<string, unknown>;
      renderScreen();
      await user.click(screen.getByText("SETTINGS$AGENT_ADVANCED_TAB"));
      const textarea = screen.getByTestId("agent-env-input");
      await user.clear(textarea);
      await user.type(textarea, "bad json");
      await waitFor(() => {
        expect(screen.getByTestId("agent-save-button")).toBeDisabled();
      });
    });
  });

  describe("save payload", () => {
    it("save button is disabled for OpenHands mode when no changes are made", () => {
      // OpenHands (kind:llm) is the default and has no API key field, so the
      // form stays clean and the save button remains disabled. This verifies
      // the component correctly represents OpenHands state.
      mockSettings.data = {
        agent_settings: { agent_kind: "llm" },
        llm_api_key_set: false,
      } as Record<string, unknown>;
      renderScreen();
      expect(screen.getByTestId("agent-save-button")).toBeDisabled();
      // No API key field — OpenHands doesn't need one via this page
      expect(
        screen.queryByTestId("agent-api-key-input"),
      ).not.toBeInTheDocument();
    });

    it("sends kind:acp with acp_server when ACP agent API key is saved", async () => {
      const user = userEvent.setup();
      mockSettings.data = {
        agent_settings: { agent_kind: "acp", acp_server: "claude-code" },
        llm_api_key_set: false,
      } as Record<string, unknown>;
      renderScreen();
      const apiKeyInput = screen.getByTestId("agent-api-key-input");
      await user.type(apiKeyInput, "sk-ant-test");
      const saveBtn = screen.getByTestId("agent-save-button");
      await waitFor(() => expect(saveBtn).not.toBeDisabled());
      await user.click(saveBtn);
      expect(mockSaveSettings).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_settings_diff: expect.objectContaining({
            agent_kind: "acp",
            acp_server: "claude-code",
            llm: { api_key: "sk-ant-test" },
          }),
        }),
        expect.any(Object),
      );
    });

    it("sends kind:acp without llm when API key is not entered", async () => {
      // When switching to ACP but not entering an API key, the diff should
      // include the ACP config but omit the llm.api_key field.
      const user = userEvent.setup();
      mockSettings.data = {
        agent_settings: { agent_kind: "acp", acp_server: "codex" },
        llm_api_key_set: false,
      } as Record<string, unknown>;
      renderScreen();
      // Navigate to Advanced tab and change command to make form dirty
      const advTab = await screen.findByText("SETTINGS$AGENT_ADVANCED_TAB");
      fireEvent.click(advTab);
      const commandInput = screen.getByTestId("agent-command-input");
      fireEvent.change(commandInput, { target: { value: "npx\ncodex-acp" } });
      const saveBtn = screen.getByTestId("agent-save-button");
      await waitFor(() => expect(saveBtn).not.toBeDisabled());
      await user.click(saveBtn);
      expect(mockSaveSettings).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_settings_diff: expect.objectContaining({
            agent_kind: "acp",
            acp_server: "codex",
            acp_command: ["npx", "codex-acp"],
          }),
        }),
        expect.any(Object),
      );
      // No llm key since no api_key was entered
      expect(
        mockSaveSettings.mock.calls[0][0].agent_settings_diff,
      ).not.toHaveProperty("llm");
    });
  });

  describe("provider switching", () => {
    it("shows correct API key label for each provider after switching", async () => {
      const user = userEvent.setup();
      mockSettings.data = {
        agent_settings: { agent_kind: "acp", acp_server: "claude-code" },
      } as Record<string, unknown>;
      renderScreen();
      // Claude Code starts with Anthropic key
      expect(screen.getByText("Anthropic API Key")).toBeInTheDocument();

      // Simulate switching to codex — the selector is an Autocomplete (HeroUI),
      // so we check the label would change once the new settings are loaded.
      // We verify by re-mounting with codex settings.
      queryClient.clear();
      mockSettings.data = {
        agent_settings: { agent_kind: "acp", acp_server: "codex" },
      } as Record<string, unknown>;
      const { unmount } = renderScreen();
      expect(screen.getAllByText("OpenAI API Key")[0]).toBeInTheDocument();
      unmount();
    });

    it("shows Google API key label for gemini-cli", () => {
      mockSettings.data = {
        agent_settings: { agent_kind: "acp", acp_server: "gemini-cli" },
      } as Record<string, unknown>;
      renderScreen();
      expect(screen.getByText("Google API Key")).toBeInTheDocument();
    });
  });

  describe("advanced tab state restoration", () => {
    it("populates command field from saved settings", async () => {
      mockSettings.data = {
        agent_settings: {
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: ["npx", "-y", "@agentclientprotocol/claude-agent-acp"],
          acp_args: [],
          acp_env: {},
        },
      } as Record<string, unknown>;
      renderScreen();
      const advTab = await screen.findByText("SETTINGS$AGENT_ADVANCED_TAB");
      fireEvent.click(advTab);
      await waitFor(() => {
        const commandArea = screen.getByTestId(
          "agent-command-input",
        ) as HTMLTextAreaElement;
        expect(commandArea.value).toBe(
          "npx\n-y\n@agentclientprotocol/claude-agent-acp",
        );
      });
    });

    it("populates env field from saved acp_env settings", async () => {
      mockSettings.data = {
        agent_settings: {
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: [],
          acp_args: [],
          acp_env: { MY_VAR: "my_value" },
        },
      } as Record<string, unknown>;
      renderScreen();
      // Wait for the Advanced tab button to appear (confirming ACP state is set)
      // before switching to it, so the useEffect has had a chance to run.
      const advancedTab = await screen.findByText(
        "SETTINGS$AGENT_ADVANCED_TAB",
      );
      fireEvent.click(advancedTab);
      await waitFor(() => {
        const envArea = screen.getByTestId(
          "agent-env-input",
        ) as HTMLTextAreaElement;
        expect(envArea.value).toContain('"MY_VAR": "my_value"');
      });
    });

    it("defaults to Basic tab on initial load", () => {
      renderScreen();
      // Basic tab button should be rendered
      expect(screen.getByText("SETTINGS$AGENT_BASIC_TAB")).toBeInTheDocument();
      // Advanced tab should not be visible when OpenHands is selected
      expect(
        screen.queryByText("SETTINGS$AGENT_ADVANCED_TAB"),
      ).not.toBeInTheDocument();
    });
  });
});

describe("buildAgentSettingsDiff", () => {
  const baseArgs = {
    command: [],
    args: [],
    parsedEnv: {},
    acpModel: "",
    initialAcpModel: "",
    apiKey: "",
    apiKeyTouched: false,
    baseUrl: "",
    initialBaseUrl: "",
  };

  it("preserves existing ACP credentials when saving unrelated changes for the same provider", () => {
    expect(
      buildAgentSettingsDiff({
        ...baseArgs,
        agentType: "claude-code",
        initialAgentType: "claude-code",
        command: ["npx", "custom"],
      }),
    ).toEqual({
      agent_kind: "acp",
      acp_server: "claude-code",
      acp_command: ["npx", "custom"],
      acp_args: [],
      acp_env: {},
    });
  });

  it("clears stale provider credentials when switching ACP providers", () => {
    expect(
      buildAgentSettingsDiff({
        ...baseArgs,
        agentType: "codex",
        initialAgentType: "claude-code",
      }),
    ).toEqual({
      agent_kind: "acp",
      acp_server: "codex",
      acp_command: [],
      acp_args: [],
      acp_env: {},
      acp_model: null,
      llm: {
        api_key: null,
        base_url: null,
      },
    });
  });

  it("clears stale ACP-only fields when switching back to OpenHands", () => {
    expect(
      buildAgentSettingsDiff({
        ...baseArgs,
        agentType: "openhands",
        initialAgentType: "gemini-cli",
      }),
    ).toEqual({
      agent_kind: "llm",
      acp_server: null,
      acp_command: null,
      acp_args: null,
      acp_env: null,
      acp_model: null,
      llm: {
        api_key: null,
        base_url: null,
      },
    });
  });

  it("sends null for cleared base URL and model fields", () => {
    expect(
      buildAgentSettingsDiff({
        ...baseArgs,
        agentType: "claude-code",
        initialAgentType: "claude-code",
        initialBaseUrl: "https://proxy.example.com",
        initialAcpModel: "claude-sonnet",
      }),
    ).toEqual({
      agent_kind: "acp",
      acp_server: "claude-code",
      acp_command: [],
      acp_args: [],
      acp_env: {},
      acp_model: null,
      llm: {
        base_url: null,
      },
    });
  });
});
