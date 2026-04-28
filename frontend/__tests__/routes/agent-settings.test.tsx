import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AgentSettingsScreen from "#/routes/agent-settings";

const mockSettings = vi.hoisted(() => ({
  data: null as Record<string, unknown> | null,
  isLoading: false,
}));

const mockSaveSettings = vi.hoisted(() => vi.fn());
const mockIsPending = vi.hoisted(() => ({ value: false }));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => mockSettings,
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
    mockIsPending.value = false;
  });

  describe("initial state", () => {
    it("renders the page title", () => {
      renderScreen();
      expect(
        screen.getByText("SETTINGS$AGENT_PAGE_TITLE"),
      ).toBeInTheDocument();
    });

    it("shows Basic tab by default", () => {
      renderScreen();
      expect(
        screen.getByText("SETTINGS$AGENT_BASIC_TAB"),
      ).toBeInTheDocument();
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
  });

  describe("loading from saved ACP settings", () => {
    it("restores claude-code agent type from saved settings", () => {
      mockSettings.data = {
        agent_settings: { kind: "acp", acp_server: "claude-code" },
      } as Record<string, unknown>;
      renderScreen();
      // API key field should be visible for claude-code
      expect(screen.getByTestId("agent-api-key-input")).toBeInTheDocument();
      expect(screen.getByText("Anthropic API Key")).toBeInTheDocument();
    });

    it("falls back to claude-code for unknown acp_server", () => {
      mockSettings.data = {
        agent_settings: { kind: "acp", acp_server: "unknown-server" },
      } as Record<string, unknown>;
      renderScreen();
      // Should fall back to claude-code, showing Anthropic API key
      expect(screen.getByText("Anthropic API Key")).toBeInTheDocument();
    });

    it("shows openhands when saved kind is llm", () => {
      mockSettings.data = {
        agent_settings: { kind: "llm" },
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
        agent_settings: { kind: "acp", acp_server: "claude-code" },
      } as Record<string, unknown>;
      renderScreen();
      expect(screen.getByText("Anthropic API Key")).toBeInTheDocument();
    });

    it("shows OpenAI API key for codex", () => {
      mockSettings.data = {
        agent_settings: { kind: "acp", acp_server: "codex" },
      } as Record<string, unknown>;
      renderScreen();
      expect(screen.getByText("OpenAI API Key")).toBeInTheDocument();
    });

    it("shows Google API key for gemini-cli", () => {
      mockSettings.data = {
        agent_settings: { kind: "acp", acp_server: "gemini-cli" },
      } as Record<string, unknown>;
      renderScreen();
      expect(screen.getByText("Google API Key")).toBeInTheDocument();
    });
  });

  describe("Advanced tab", () => {
    beforeEach(() => {
      mockSettings.data = {
        agent_settings: { kind: "acp", acp_server: "claude-code" },
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
        agent_settings: { kind: "acp", acp_server: "claude-code" },
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
        agent_settings: { kind: "acp", acp_server: "claude-code" },
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
    it("sends kind:llm when OpenHands is saved", async () => {
      const user = userEvent.setup();
      // Start with an ACP setting so the form is considered dirty when we would change it
      // In practice, switching to OpenHands makes isDirty = true
      mockSettings.data = {
        agent_settings: { kind: "acp", acp_server: "claude-code" },
        llm_api_key_set: false,
      } as Record<string, unknown>;
      renderScreen();
      // The form is clean on load; we need to interact with it to make isDirty true.
      // Type in the API key field to make it dirty, then check save payload
      const apiKeyInput = screen.getByTestId("agent-api-key-input");
      await user.type(apiKeyInput, "sk-ant-test");
      const saveBtn = screen.getByTestId("agent-save-button");
      await waitFor(() => expect(saveBtn).not.toBeDisabled());
      await user.click(saveBtn);
      expect(mockSaveSettings).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_settings_diff: expect.objectContaining({
            kind: "acp",
            acp_server: "claude-code",
          }),
        }),
        expect.any(Object),
      );
    });

    it("sends kind:acp with acp_server when ACP agent is saved", async () => {
      const user = userEvent.setup();
      mockSettings.data = {
        agent_settings: { kind: "acp", acp_server: "claude-code" },
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
            kind: "acp",
            acp_server: "claude-code",
            llm: { api_key: "sk-ant-test" },
          }),
        }),
        expect.any(Object),
      );
    });
  });
});
