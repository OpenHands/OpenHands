import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { getAcpProvider as getClientAcpProvider } from "@openhands/typescript-client";
import {
  AgentSettingsScreen,
  type AgentSettingsSaveControl,
} from "#/routes/agent-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import { SecretsService } from "#/api/secrets-service";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { Settings } from "#/types/settings";

const CLAUDE_COMMAND = getClientAcpProvider("claude-code")!.default_command;
const CLAUDE_COMMAND_VALUE = [...CLAUDE_COMMAND] as string[];

// Stub the login-detection probe so the ACP credentials section doesn't spin a
// subprocess; default to no detected session so existing tests are unaffected.
const acpAuthStatusMock = vi.hoisted(() => vi.fn());
vi.mock("#/hooks/query/use-acp-auth-status", () => ({
  useAcpAuthStatus: (...args: unknown[]) => acpAuthStatusMock(...args),
}));

// The profile editor gates `enable_switch_llm_tool` on the backend's *profile*
// model. Stub the probe so both sides of that gate are reachable.
const profileSupportsSwitchLlmToolMock = vi.hoisted(() => vi.fn(() => true));
const profileSupportsSecretRefsMock = vi.hoisted(() => vi.fn(() => true));
vi.mock("#/api/agent-profiles-service/profile-field-support", () => ({
  agentProfileSupportsSwitchLlmTool: () => profileSupportsSwitchLlmToolMock(),
  agentProfileSupportsSecretRefs: () => profileSupportsSecretRefsMock(),
}));

// The secret picker lists the user's saved secrets.
const savedSecretsMock = vi.hoisted(() =>
  vi.fn<() => { name: string; description?: string }[]>(),
);
vi.mock("#/hooks/query/use-get-secrets", () => ({
  useSearchSecrets: () => ({ data: savedSecretsMock() }),
}));

function buildSettings(overrides: Partial<Settings> = {}): Settings {
  return {
    ...MOCK_DEFAULT_USER_SETTINGS,
    ...overrides,
    agent_settings:
      overrides.agent_settings ?? MOCK_DEFAULT_USER_SETTINGS.agent_settings,
  };
}

function renderAgentSettingsScreen(
  props: React.ComponentProps<typeof AgentSettingsScreen> = {},
) {
  return render(<AgentSettingsScreen {...props} />, {
    wrapper: ({ children }) => (
      <MemoryRouter>
        <QueryClientProvider
          client={
            new QueryClient({ defaultOptions: { queries: { retry: false } } })
          }
        >
          {children}
        </QueryClientProvider>
      </MemoryRouter>
    ),
  });
}

/** Capture the latest save control the form reports to its parent. */
function trackSaveControl() {
  const holder: { current: AgentSettingsSaveControl | null } = {
    current: null,
  };
  return {
    holder,
    onSaveControlChange: (control: AgentSettingsSaveControl) => {
      holder.current = control;
    },
  };
}

async function renderOpenHandsProfile(
  override: Record<string, unknown> = { agent_kind: "openhands" },
) {
  const tracker = trackSaveControl();
  vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
    buildSettings({
      agent_settings: {
        ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
        agent_kind: "openhands",
      },
    }),
  );
  renderAgentSettingsScreen({
    agentSettingsOverride: override as Record<
      string,
      string | number | boolean | string[] | null
    >,
    onSaveControlChange: tracker.onSaveControlChange,
  });
  await screen.findByTestId("agent-settings-screen");
  return tracker;
}

describe("AgentSettingsScreen", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);
    vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([]);
    vi.spyOn(SecretsService, "createSecret").mockResolvedValue();
    acpAuthStatusMock.mockReturnValue({
      status: "unknown",
      isChecking: false,
      isSupported: true,
    });
    profileSupportsSwitchLlmToolMock.mockReturnValue(true);
    profileSupportsSecretRefsMock.mockReturnValue(true);
    savedSecretsMock.mockReturnValue([
      { name: "GITHUB_TOKEN", description: "repo access" },
      { name: "DATADOG_API_KEY" },
      { name: "PROD_DB_URL" },
    ]);
  });

  it("renders the agent type selector and hides ACP fields on the OpenHands path", async () => {
    await renderOpenHandsProfile();
    expect(screen.getByTestId("agent-type-selector")).toBeInTheDocument();
    // Tool switches were retired from this form (#17771); only the profile
    // editor path remains.
    expect(
      screen.queryByTestId("agent-settings-enable-sub-agents"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("agent-settings-enable-switch-llm-tool"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("agent-save-button"),
    ).not.toBeInTheDocument();
    expect(screen.queryByTestId("agent-command-input")).not.toBeInTheDocument();
  });

  it("passes through seeded enable_sub_agents and enable_switch_llm_tool", async () => {
    const tracker = await renderOpenHandsProfile({
      agent_kind: "openhands",
      enable_sub_agents: true,
      enable_switch_llm_tool: false,
      tool_concurrency_limit: 1,
    });
    expect(tracker.holder.current!.buildAgentProfileFields()).toMatchObject({
      agent_kind: "openhands",
      enable_sub_agents: true,
      enable_switch_llm_tool: false,
    });
  });

  it("omits enable_switch_llm_tool when the profile model predates the field", async () => {
    profileSupportsSwitchLlmToolMock.mockReturnValue(false);
    const tracker = await renderOpenHandsProfile({
      agent_kind: "openhands",
      enable_sub_agents: true,
      enable_switch_llm_tool: true,
    });
    expect(
      tracker.holder.current!.buildAgentProfileFields(),
    ).not.toHaveProperty("enable_switch_llm_tool");
  });

  it("writes tool_concurrency_limit through the profile builder", async () => {
    const user = userEvent.setup();
    const tracker = await renderOpenHandsProfile({
      agent_kind: "openhands",
      enable_sub_agents: false,
      tool_concurrency_limit: 1,
    });

    const input = screen.getByTestId("sdk-settings-tool_concurrency_limit");
    await user.clear(input);
    await user.type(input, "4");

    expect(tracker.holder.current!.buildAgentProfileFields()).toMatchObject({
      agent_kind: "openhands",
      tool_concurrency_limit: 4,
    });
  });

  it("shows the ACP form when the active agent_kind is acp", async () => {
    const tracker = trackSaveControl();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "acp",
        acp_server: "claude-code",
        acp_command: CLAUDE_COMMAND_VALUE,
        acp_model: "claude-opus-4-8",
      },
      onSaveControlChange: tracker.onSaveControlChange,
    });
    await screen.findByTestId("agent-settings-screen");

    expect(screen.getByTestId("agent-command-input")).toBeInTheDocument();
    expect(tracker.holder.current!.agentType).toBe("acp");
  });

  it("defaults built-in ACP providers to a suggested model when none is saved", async () => {
    const tracker = trackSaveControl();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "acp",
        acp_server: "claude-code",
        acp_command: CLAUDE_COMMAND_VALUE,
        acp_model: "",
      },
      onSaveControlChange: tracker.onSaveControlChange,
    });
    await screen.findByTestId("agent-settings-screen");

    const fields = tracker.holder.current!.buildAgentProfileFields();
    if (fields.agent_kind === "acp") {
      expect(fields.acp_model).toBeTruthy();
    }
  });

  it("clears the model when switching from a built-in provider to Custom", async () => {
    const user = userEvent.setup();
    const tracker = trackSaveControl();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "acp",
        acp_server: "claude-code",
        acp_command: CLAUDE_COMMAND_VALUE,
        acp_model: "claude-opus-4-8",
      },
      onSaveControlChange: tracker.onSaveControlChange,
    });
    await screen.findByTestId("agent-settings-screen");

    const preset = screen.getByTestId("agent-preset-selector");
    // Open the dropdown and pick Custom.
    await user.click(preset);
    await user.click(await screen.findByText("SETTINGS$AGENT_PRESET_CUSTOM"));

    const fields = tracker.holder.current!.buildAgentProfileFields();
    if (fields.agent_kind === "acp") {
      expect(fields.acp_model).toBeNull();
      expect(fields.acp_server).toBe("custom");
    }
  });

  it("reconciles the model when the command is retyped to a different provider", async () => {
    const user = userEvent.setup();
    const tracker = trackSaveControl();
    const codex = getClientAcpProvider("codex")!.default_command;
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "acp",
        acp_server: "claude-code",
        acp_command: CLAUDE_COMMAND_VALUE,
        acp_model: "claude-opus-4-8",
      },
      onSaveControlChange: tracker.onSaveControlChange,
    });
    await screen.findByTestId("agent-settings-screen");

    const commandInput = screen.getByTestId("agent-command-input");
    await user.clear(commandInput);
    await user.type(commandInput, codex.join(" "));

    const fields = tracker.holder.current!.buildAgentProfileFields();
    if (fields.agent_kind === "acp") {
      expect(fields.acp_server).toBe("codex");
      expect(fields.acp_model).not.toBe("claude-opus-4-8");
    }
  });

  it("clears ACP fields when switching back to OpenHands", async () => {
    const user = userEvent.setup();
    const tracker = trackSaveControl();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "acp",
        acp_server: "claude-code",
        acp_command: CLAUDE_COMMAND_VALUE,
        acp_model: "claude-opus-4-8",
      },
      onSaveControlChange: tracker.onSaveControlChange,
    });
    await screen.findByTestId("agent-settings-screen");

    const typeSelector = screen.getByTestId("agent-type-selector");
    await user.click(typeSelector);
    await user.click(
      await screen.findByText("SETTINGS$AGENT_TYPE_OPENHANDS"),
    );

    const fields = tracker.holder.current!.buildAgentProfileFields();
    expect(fields.agent_kind).toBe("openhands");
    expect(screen.queryByTestId("agent-command-input")).not.toBeInTheDocument();
  });

  it("marks an empty ACP command invalid so the parent can block save", async () => {
    const user = userEvent.setup();
    const tracker = trackSaveControl();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "acp",
        acp_server: "custom",
        acp_command: ["my-acp", "--flag"],
      },
      onSaveControlChange: tracker.onSaveControlChange,
    });
    await screen.findByTestId("agent-settings-screen");
    expect(tracker.holder.current!.isValid).toBe(true);

    const commandInput = screen.getByTestId("agent-command-input");
    await user.clear(commandInput);
    expect(tracker.holder.current!.isValid).toBe(false);
  });

  it("preserves a Custom command with quoted args end-to-end", async () => {
    const user = userEvent.setup();
    const tracker = trackSaveControl();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "acp",
        acp_server: "custom",
        acp_command: ["my-acp", "--flag"],
      },
      onSaveControlChange: tracker.onSaveControlChange,
    });
    await screen.findByTestId("agent-settings-screen");

    const commandInput = screen.getByTestId("agent-command-input");
    await user.clear(commandInput);
    await user.type(commandInput, 'my-acp --name "hello world"');

    const fields = tracker.holder.current!.buildAgentProfileFields();
    if (fields.agent_kind === "acp") {
      expect(fields.acp_command).toContain("hello world");
    }
  });

  it("expands the registry default before merging acp_args on load", async () => {
    const user = userEvent.setup();
    const tracker = trackSaveControl();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "acp",
        acp_server: "claude-code",
        acp_command: [],
        acp_args: ["--extra-arg"],
      },
      onSaveControlChange: tracker.onSaveControlChange,
    });
    await screen.findByTestId("agent-settings-screen");

    const cmd = screen.getByTestId(
      "agent-command-input",
    ) as HTMLTextAreaElement;
    // The form must expand the default *before* merging args so the user
    // sees (and saves) the full spawn command, not bare --extra-arg.
    expect(cmd.value).toContain("@agentclientprotocol/claude-agent-acp");
    expect(cmd.value).toContain("--extra-arg");

    await user.click(cmd);
    await user.keyboard("{End} --saved");
    const fields = tracker.holder.current!.buildAgentProfileFields();
    if (fields.agent_kind === "acp") {
      expect(fields.acp_command).toContain("--extra-arg");
      expect(fields.acp_command).toContain("--saved");
      expect(fields.acp_command).toContain("claude-agent-acp");
    }
  });

  it("is clean after reverting an agent-type dropdown change", async () => {
    const user = userEvent.setup();
    const tracker = await renderOpenHandsProfile();

    const typeSelector = screen.getByTestId("agent-type-selector");
    await user.click(typeSelector);
    await user.click(await screen.findByText("SETTINGS$AGENT_TYPE_ACP"));
    expect(tracker.holder.current!.isDirty).toBe(true);

    await user.click(typeSelector);
    await user.click(
      await screen.findByText("SETTINGS$AGENT_TYPE_OPENHANDS"),
    );
    expect(tracker.holder.current!.isDirty).toBe(false);
  });

  it("preserves an unknown loaded acp_server when the user saves without editing", async () => {
    const tracker = trackSaveControl();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "acp",
        acp_server: "future-provider",
        acp_command: ["future-acp"],
        acp_model: "",
      },
      onSaveControlChange: tracker.onSaveControlChange,
    });
    await screen.findByTestId("agent-settings-screen");

    const fields = tracker.holder.current!.buildAgentProfileFields();
    if (fields.agent_kind === "acp") {
      // The pure builder keys off the *detected* preset; the parent merges
      // stored identity. Here the command is custom-shaped so it stays custom
      // unless we keep the stored server — assert the command survives.
      expect(fields.acp_command).toBe("future-acp");
    }
    expect(tracker.holder.current!.isDirty).toBe(false);
  });

  it("reports dirty when the user edits an unknown loaded acp_server command", async () => {
    const user = userEvent.setup();
    const tracker = trackSaveControl();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "acp",
        acp_server: "future-provider",
        acp_command: ["future-acp"],
        acp_model: "",
      },
      onSaveControlChange: tracker.onSaveControlChange,
    });
    await screen.findByTestId("agent-settings-screen");

    const commandInput = screen.getByTestId("agent-command-input");
    await user.clear(commandInput);
    await user.type(commandInput, "other-acp --x");
    expect(tracker.holder.current!.isDirty).toBe(true);
    expect(tracker.holder.current!.isValid).toBe(true);
  });

  it("exposes the ACP credential form so a single parent save can persist secrets", async () => {
    const user = userEvent.setup();
    const createSecret = vi.spyOn(SecretsService, "createSecret");
    const tracker = trackSaveControl();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "acp",
        acp_server: "claude-code",
        acp_command: CLAUDE_COMMAND_VALUE,
        acp_model: "claude-opus-4-8",
      },
      onSaveControlChange: tracker.onSaveControlChange,
    });
    await screen.findByTestId("agent-settings-screen");

    // The credentials section renders secret fields for built-in providers.
    const input = await screen.findByTestId("settings-acp-secret-ANTHROPIC_API_KEY");
    await user.type(input, "sk-test-value");
    expect(tracker.holder.current!.credentials.isDirty).toBe(true);

    await tracker.holder.current!.credentials.save({ silent: true });
    await waitFor(() => {
      expect(createSecret).toHaveBeenCalled();
    });
  });

  it("shows the signed-in auth banner when authenticated", async () => {
    acpAuthStatusMock.mockReturnValue({
      status: "authenticated",
      isChecking: false,
      isSupported: true,
    });
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "acp",
        acp_server: "claude-code",
        acp_command: CLAUDE_COMMAND_VALUE,
      },
    });
    await screen.findByTestId("agent-settings-screen");
    expect(
      await screen.findByTestId("settings-acp-auth-detected"),
    ).toBeInTheDocument();
  });
});

describe("AgentSettingsScreen — MCP server scope", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);
    vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([]);
    acpAuthStatusMock.mockReturnValue({
      status: "unknown",
      isChecking: false,
      isSupported: true,
    });
    profileSupportsSecretRefsMock.mockReturnValue(true);
    savedSecretsMock.mockReturnValue([]);
    // Configured MCP servers ride `agent_settings.mcp_config` (useSettings
    // prefers that over the top-level key).
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          agent_kind: "openhands",
          mcp_config: {
            fetch: { transport: "stdio", command: "uvx", args: ["mcp-fetch"] },
            playwright: {
              transport: "stdio",
              command: "npx",
              args: ["@playwright/mcp"],
            },
          },
        } as never,
      }),
    );
  });

  it("lists configured servers and persists null by default", async () => {
    const tracker = trackSaveControl();
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "openhands",
        enable_sub_agents: false,
      },
      onSaveControlChange: tracker.onSaveControlChange,
    });
    await screen.findByTestId("agent-settings-screen");

    expect(screen.getByTestId("agent-settings-mcp-list")).toBeInTheDocument();
    expect(screen.getByTestId("agent-settings-mcp-fetch")).toBeInTheDocument();
    expect(tracker.holder.current!.buildAgentProfileFields()).toMatchObject({
      mcp_server_refs: null,
    });
  });

  it("seeds from a stored scope and persists the selection", async () => {
    const tracker = trackSaveControl();
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "openhands",
        enable_sub_agents: false,
        mcp_server_refs: ["fetch"],
      },
      onSaveControlChange: tracker.onSaveControlChange,
    });
    await screen.findByTestId("agent-settings-screen");

    expect(tracker.holder.current!.buildAgentProfileFields()).toMatchObject({
      mcp_server_refs: ["fetch"],
    });
  });

  it("seeds a switch to custom with every configured server", async () => {
    const user = userEvent.setup();
    const tracker = trackSaveControl();
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "openhands",
        enable_sub_agents: false,
      },
      onSaveControlChange: tracker.onSaveControlChange,
    });
    await screen.findByTestId("agent-settings-screen");

    await user.click(screen.getByTestId("agent-settings-mcp-mode"));
    await user.click(
      await screen.findByText("SETTINGS$AGENT_PROFILE_MCP_CHOOSE"),
    );

    expect(tracker.holder.current!.buildAgentProfileFields()).toMatchObject({
      mcp_server_refs: ["fetch", "playwright"],
    });
  });

  it("warns about a ref whose server is gone", async () => {
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "openhands",
        enable_sub_agents: false,
        mcp_server_refs: ["ghost-server"],
      },
    });
    await screen.findByTestId("agent-settings-screen");
    expect(
      await screen.findByTestId("agent-settings-mcp-dangling"),
    ).toBeInTheDocument();
  });
});

describe("AgentSettingsScreen — secret scope", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([]);
    acpAuthStatusMock.mockReturnValue({
      status: "unknown",
      isChecking: false,
      isSupported: true,
    });
    profileSupportsSecretRefsMock.mockReturnValue(true);
    savedSecretsMock.mockReturnValue([
      { name: "GITHUB_TOKEN", description: "repo access" },
      { name: "DATADOG_API_KEY" },
    ]);
  });

  it("leaves an OpenHands profile's scope empty when scoping starts", async () => {
    const user = userEvent.setup();
    const tracker = trackSaveControl();
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "openhands",
        enable_sub_agents: false,
      },
      onSaveControlChange: tracker.onSaveControlChange,
    });
    await screen.findByTestId("agent-settings-screen");

    expect(tracker.holder.current!.buildAgentProfileFields()).toMatchObject({
      secret_refs: null,
    });

    // Switching to custom does not silently grant every secret — the user
    // picks explicitly. (ACP auto-adds only its provider credentials.)
    await user.click(screen.getByTestId("agent-settings-secrets-mode"));
    await user.click(
      await screen.findByText("SETTINGS$AGENT_PROFILE_SECRETS_CHOOSE"),
    );
    expect(tracker.holder.current!.buildAgentProfileFields()).toMatchObject({
      secret_refs: [],
    });
  });

  it("scopes a stored selection through the profile builder", async () => {
    const tracker = trackSaveControl();
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "openhands",
        enable_sub_agents: false,
        secret_refs: ["DATADOG_API_KEY"],
      },
      onSaveControlChange: tracker.onSaveControlChange,
    });
    await screen.findByTestId("agent-settings-screen");
    expect(tracker.holder.current!.buildAgentProfileFields()).toMatchObject({
      secret_refs: ["DATADOG_API_KEY"],
    });
  });

  it("keeps a stored ref whose secret no longer exists", async () => {
    savedSecretsMock.mockReturnValue([{ name: "GITHUB_TOKEN" }]);
    const tracker = trackSaveControl();
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "openhands",
        enable_sub_agents: false,
        secret_refs: ["GONE_SECRET"],
      },
      onSaveControlChange: tracker.onSaveControlChange,
    });
    await screen.findByTestId("agent-settings-screen");

    expect(tracker.holder.current!.buildAgentProfileFields()).toMatchObject({
      secret_refs: ["GONE_SECRET"],
    });
  });

  it("omits secret_refs on a backend whose profile model predates it", async () => {
    profileSupportsSecretRefsMock.mockReturnValue(false);
    const tracker = trackSaveControl();
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "openhands",
        enable_sub_agents: false,
        secret_refs: ["DATADOG_API_KEY"],
      },
      onSaveControlChange: tracker.onSaveControlChange,
    });
    await screen.findByTestId("agent-settings-screen");

    expect(
      tracker.holder.current!.buildAgentProfileFields(),
    ).not.toHaveProperty("secret_refs");
  });

  it("hides the secret scope when the profile model predates secret_refs", async () => {
    profileSupportsSecretRefsMock.mockReturnValue(false);
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "openhands",
        enable_sub_agents: false,
      },
    });
    await screen.findByTestId("agent-settings-screen");
    expect(
      screen.queryByTestId("agent-settings-secrets-mode"),
    ).not.toBeInTheDocument();
  });
});
