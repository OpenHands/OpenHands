import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
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
import { ACP_PROVIDERS } from "#/constants/acp-providers";
import { parseCommand } from "#/utils/acp-command";
const CLAUDE_COMMAND = getClientAcpProvider("claude-code")!.default_command;
const CODEX_COMMAND = getClientAcpProvider("codex")!.default_command;

// Stub the login-detection probe so the ACP credentials section doesn't spin a
// subprocess; default to no detected session so existing tests are unaffected.
const acpAuthStatusMock = vi.hoisted(() => vi.fn());
vi.mock("#/hooks/query/use-acp-auth-status", () => ({
  useAcpAuthStatus: (...args: unknown[]) => acpAuthStatusMock(...args),
}));

// The LLM-switching toggle is gated on the backend's *profile* model, which
// gained the field later than the settings schema did. Stub the probe so both
// sides of that gate are reachable without a live server.
const profileSupportsSwitchLlmToolMock = vi.hoisted(() => vi.fn(() => true));
const profileSupportsSecretRefsMock = vi.hoisted(() => vi.fn(() => true));
const profileSupportsSystemPromptMock = vi.hoisted(() => vi.fn(() => false));
const profileSupportsInstructionsMock = vi.hoisted(() => vi.fn(() => false));
vi.mock("#/api/agent-profiles-service/profile-field-support", () => ({
  agentProfileSupportsSwitchLlmTool: () => profileSupportsSwitchLlmToolMock(),
  agentProfileSupportsSecretRefs: () => profileSupportsSecretRefsMock(),
  agentProfileSupportsSystemPrompt: () => profileSupportsSystemPromptMock(),
  agentProfileSupportsInstructions: () => profileSupportsInstructionsMock(),
}));

// The secret picker lists the user's saved secrets; stub the query so these
// tests don't need a live secrets store.
const savedSecretsMock = vi.hoisted(() =>
  vi.fn<() => { name: string; description?: string }[]>(),
);
vi.mock("#/hooks/query/use-get-secrets", () => ({
  useSearchSecrets: () => ({ data: savedSecretsMock() }),
}));

// Observe toasts so a silent credential save can be asserted as silent.
const toastMocks = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  warning: vi.fn(),
}));
vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: toastMocks.success,
  displayErrorToast: toastMocks.error,
  displayWarningToast: toastMocks.warning,
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
  props: Partial<React.ComponentProps<typeof AgentSettingsScreen>> = {},
) {
  let control: AgentSettingsSaveControl | null = null;
  const view = render(
    <AgentSettingsScreen
      onSaveControlChange={(next) => {
        control = next;
      }}
      {...props}
    />,
    {
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
    },
  );
  return { ...view, control: () => control! };
}

const CLAUDE_PROFILE = {
  agent_kind: "acp",
  acp_server: "claude-code",
  acp_command: [],
  acp_args: [],
  acp_model: "",
};

describe("AgentSettingsScreen", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    // The form owns the ACP credential form, which reads secrets on mount.
    vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([]);
    vi.spyOn(SecretsService, "createSecret").mockResolvedValue();
    acpAuthStatusMock.mockReturnValue({
      status: "unknown",
      isChecking: false,
      isSupported: true,
    });
    toastMocks.success.mockClear();
    toastMocks.error.mockClear();
    toastMocks.warning.mockClear();
    profileSupportsSwitchLlmToolMock.mockReturnValue(true);
    profileSupportsSecretRefsMock.mockReturnValue(true);
    savedSecretsMock.mockReturnValue([
      { name: "GITHUB_TOKEN", description: "repo access" },
      { name: "DATADOG_API_KEY" },
      { name: "PROD_DB_URL" },
    ]);
  });

  it("renders the agent type selector defaulting to OpenHands with sub-agents toggle", async () => {
    renderAgentSettingsScreen({
      agentSettingsOverride: { agent_kind: "openhands" },
    });
    await screen.findByTestId("agent-settings-screen");

    expect(screen.getByLabelText("SETTINGS$NAV_AGENT")).toHaveValue(
      "SETTINGS$AGENT_TYPE_OPENHANDS",
    );
    expect(
      screen.getByTestId("agent-settings-enable-sub-agents"),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("agent-command-input")).not.toBeInTheDocument();
  });

  it("builds enable_sub_agents from its toggle", async () => {
    const user = userEvent.setup();
    const { control } = renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "openhands",
        enable_sub_agents: false,
      },
    });
    await screen.findByTestId("agent-settings-screen");

    const toggle = screen.getByTestId("agent-settings-enable-sub-agents");
    await user.click(toggle.closest("label")!);

    expect(control().buildAgentProfileFields()).toMatchObject({
      enable_sub_agents: true,
    });
  });

  it("builds enable_switch_llm_tool from its toggle", async () => {
    const user = userEvent.setup();
    const { control } = renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "openhands",
        enable_switch_llm_tool: true,
      },
    });
    await screen.findByTestId("agent-settings-screen");

    const toggle = screen.getByTestId("agent-settings-enable-switch-llm-tool");
    expect(toggle).toBeChecked();
    await user.click(toggle.closest("label")!);

    expect(toggle).not.toBeChecked();
    expect(control().buildAgentProfileFields()).toMatchObject({
      enable_switch_llm_tool: false,
    });
  });

  it("hides the LLM-switching toggle when the schema predates the field", async () => {
    const schema = MOCK_DEFAULT_USER_SETTINGS.agent_settings_schema;
    const schemaWithoutField = schema && {
      ...schema,
      sections: schema.sections.map((section) => ({
        ...section,
        fields: section.fields.filter(
          (field) => field.key !== "enable_switch_llm_tool",
        ),
      })),
    };
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ agent_settings_schema: schemaWithoutField }),
    );

    renderAgentSettingsScreen({
      agentSettingsOverride: { agent_kind: "openhands" },
    });
    await screen.findByTestId("agent-settings-screen");

    // Older agent-servers without the field hide the toggle cleanly...
    expect(
      screen.queryByTestId("agent-settings-enable-switch-llm-tool"),
    ).not.toBeInTheDocument();
    // ...while the other OpenHands controls still render.
    expect(
      screen.getByTestId("agent-settings-enable-sub-agents"),
    ).toBeInTheDocument();
  });

  it("hides the LLM-switching toggle when the profile model predates the field", async () => {
    // agent-server 1.29.0–1.30.x advertises the field in the settings schema
    // while `OpenHandsAgentProfile` still rejects it. Rendering the toggle
    // there would offer a control whose save the server refuses outright.
    profileSupportsSwitchLlmToolMock.mockReturnValue(false);

    renderAgentSettingsScreen({
      agentSettingsOverride: { agent_kind: "openhands" },
    });
    await screen.findByTestId("agent-settings-screen");

    expect(
      screen.queryByTestId("agent-settings-enable-switch-llm-tool"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("agent-settings-enable-sub-agents"),
    ).toBeInTheDocument();
  });

  it("builds tool_concurrency_limit from the input on the OpenHands path", async () => {
    const user = userEvent.setup();
    const { control } = renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "openhands",
        tool_concurrency_limit: 1,
      },
    });
    await screen.findByTestId("agent-settings-screen");

    const input = screen.getByTestId("sdk-settings-tool_concurrency_limit");
    await user.clear(input);
    await user.type(input, "4");

    // Coerced to a number (not the raw input string) via the shared
    // schema-driven coercion.
    expect(control().buildAgentProfileFields()).toMatchObject({
      tool_concurrency_limit: 4,
    });
  });

  it("hides the sub-agents toggle when ACP is selected", async () => {
    const user = userEvent.setup();
    renderAgentSettingsScreen({
      agentSettingsOverride: { agent_kind: "openhands" },
    });
    await screen.findByTestId("agent-settings-screen");
    expect(
      screen.getByTestId("agent-settings-enable-sub-agents"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("agent-type-selector"));
    await user.click(
      await screen.findByRole("option", { name: "SETTINGS$AGENT_TYPE_ACP" }),
    );

    expect(
      screen.queryByTestId("agent-settings-enable-sub-agents"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("agent-command-input")).toBeInTheDocument();
  });

  it("opens a stored ACP profile on its command and custom model", async () => {
    // A model ID outside the provider's suggestions falls through to the
    // custom input; known IDs go through the dropdown instead.
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        ...CLAUDE_PROFILE,
        acp_command: ["npx", "-y", "@agentclientprotocol/claude-agent-acp"],
        acp_model: "my-pinned-fork-model",
      },
    });

    const commandInput = (await screen.findByTestId(
      "agent-command-input",
    )) as HTMLTextAreaElement;
    expect(commandInput.value).toBe(
      "npx -y @agentclientprotocol/claude-agent-acp",
    );
    const modelInput = screen.getByTestId(
      "agent-model-input",
    ) as HTMLInputElement;
    expect(modelInput.value).toBe("my-pinned-fork-model");
  });

  it("defaults built-in ACP providers to a suggested model when none is saved", async () => {
    renderAgentSettingsScreen({ agentSettingsOverride: CLAUDE_PROFILE });

    await screen.findByTestId("agent-command-input");
    expect(screen.getByLabelText("SETTINGS$AGENT_MODEL")).toHaveValue(
      "Claude Opus (1M)",
    );
  });

  it("builds the selected built-in ACP model", async () => {
    const user = userEvent.setup();
    const { control } = renderAgentSettingsScreen({
      agentSettingsOverride: CLAUDE_PROFILE,
    });

    await screen.findByTestId("agent-command-input");
    await user.click(screen.getByLabelText("SETTINGS$AGENT_MODEL"));
    await user.click(await screen.findByText("Claude Haiku"));

    expect(control().buildAgentProfileFields()).toMatchObject({
      acp_model: "haiku",
    });
  });

  it("clears the model when switching from a built-in provider to Custom", async () => {
    // Picking Custom must not leak the built-in default model onto an
    // unrelated wrapper.
    const user = userEvent.setup();
    const { control } = renderAgentSettingsScreen({
      agentSettingsOverride: CLAUDE_PROFILE,
    });
    await screen.findByTestId("agent-command-input");
    expect(screen.getByLabelText("SETTINGS$AGENT_MODEL")).toHaveValue(
      "Claude Opus (1M)",
    );

    await user.click(screen.getByTestId("agent-preset-selector"));
    await user.click(
      await screen.findByRole("option", {
        name: "SETTINGS$AGENT_PRESET_CUSTOM",
      }),
    );
    const commandInput = screen.getByTestId("agent-command-input");
    await user.clear(commandInput);
    await user.type(commandInput, "my-custom-acp --flag");

    expect(control().buildAgentProfileFields()).toMatchObject({
      acp_server: "custom",
      acp_command: "my-custom-acp --flag",
      acp_model: null,
    });
  });

  it("reconciles the model when the command is retyped to a different provider", async () => {
    // The textarea is the other way to change provider, so it must drop the
    // previous provider's model just like the preset dropdown does.
    const user = userEvent.setup();
    const { control } = renderAgentSettingsScreen({
      agentSettingsOverride: CLAUDE_PROFILE,
    });
    await screen.findByTestId("agent-command-input");

    const commandInput = screen.getByTestId("agent-command-input");
    await user.clear(commandInput);
    await user.type(commandInput, CODEX_COMMAND.join(" "));

    expect(screen.getByLabelText("SETTINGS$AGENT_MODEL")).toHaveValue(
      "GPT-5.5",
    );
    expect(control().buildAgentProfileFields()).toMatchObject({
      acp_server: "codex",
      acp_command: null,
      acp_model: "gpt-5.5",
    });
  });

  it("prefills Claude Code when switching to ACP", async () => {
    const user = userEvent.setup();
    const { control } = renderAgentSettingsScreen({
      agentSettingsOverride: { agent_kind: "openhands" },
    });
    await screen.findByTestId("agent-settings-screen");

    await user.click(screen.getByTestId("agent-type-selector"));
    await user.click(
      await screen.findByRole("option", { name: "SETTINGS$AGENT_TYPE_ACP" }),
    );

    const commandInput = (await screen.findByTestId(
      "agent-command-input",
    )) as HTMLTextAreaElement;
    expect(commandInput.value).toBe(CLAUDE_COMMAND.join(" "));
    expect(control().buildAgentProfileFields()).toEqual({
      agent_kind: "acp",
      mcp_server_refs: null,
      secret_refs: null,
      acp_server: "claude-code",
      // The default command is left to the registry rather than pinned.
      acp_command: null,
      acp_args: null,
      acp_model: "opus[1m]",
    });
  });

  it("drops the ACP fields when switching back to OpenHands", async () => {
    const user = userEvent.setup();
    const { control } = renderAgentSettingsScreen({
      agentSettingsOverride: CLAUDE_PROFILE,
    });
    await screen.findByTestId("agent-command-input");

    await user.click(screen.getByTestId("agent-type-selector"));
    await user.click(
      await screen.findByRole("option", {
        name: "SETTINGS$AGENT_TYPE_OPENHANDS",
      }),
    );

    expect(control().buildAgentProfileFields()).toEqual({
      agent_kind: "openhands",
      mcp_server_refs: null,
      secret_refs: null,
      enable_sub_agents: false,
      enable_switch_llm_tool: true,
      tool_concurrency_limit: 1,
    });
  });

  it.each(["", "   \t   "])(
    "is invalid when the ACP command is %j",
    async (command) => {
      // The agent-server cannot spawn an empty command.
      const user = userEvent.setup();
      const { control } = renderAgentSettingsScreen({
        agentSettingsOverride: CLAUDE_PROFILE,
      });
      const cmd = await screen.findByTestId("agent-command-input");
      expect(control().isValid).toBe(true);

      await user.clear(cmd);
      if (command) await user.type(cmd, command);

      expect(control().isValid).toBe(false);
    },
  );

  it("round-trips a Custom command with quoted args", async () => {
    // ``bash -c "echo hi"`` must not tokenise as
    // ``["bash","-c","\"echo","hi\""]``.
    const user = userEvent.setup();
    const { control } = renderAgentSettingsScreen({
      agentSettingsOverride: CLAUDE_PROFILE,
    });
    const cmd = await screen.findByTestId("agent-command-input");
    await user.clear(cmd);
    await user.type(cmd, 'bash -c "echo hi"');

    const fields = control().buildAgentProfileFields();
    expect(fields).toMatchObject({ acp_server: "custom" });
    expect(
      parseCommand((fields as { acp_command: string }).acp_command),
    ).toEqual(["bash", "-c", "echo hi"]);
  });

  it("keeps the registry default when a profile stores only acp_args", async () => {
    // The default command must be expanded before the args are appended, or
    // an edit would save the bare args and drop the provider's command.
    const user = userEvent.setup();
    const { control } = renderAgentSettingsScreen({
      agentSettingsOverride: { ...CLAUDE_PROFILE, acp_args: ["--extra-arg"] },
    });
    const cmd = (await screen.findByTestId(
      "agent-command-input",
    )) as HTMLTextAreaElement;
    expect(cmd.value).toBe(`${CLAUDE_COMMAND.join(" ")} --extra-arg`);

    await user.click(cmd);
    await user.keyboard("{End} --saved");

    const fields = control().buildAgentProfileFields();
    expect(fields).toMatchObject({ acp_server: "custom", acp_args: null });
    expect(
      parseCommand((fields as { acp_command: string }).acp_command),
    ).toEqual([...CLAUDE_COMMAND, "--extra-arg", "--saved"]);
  });

  it("is clean again after reverting an agent-type change", async () => {
    const user = userEvent.setup();
    const { control } = renderAgentSettingsScreen({
      agentSettingsOverride: { agent_kind: "openhands" },
    });
    await screen.findByTestId("agent-type-selector");
    expect(control().isDirty).toBe(false);

    await user.click(screen.getByTestId("agent-type-selector"));
    await user.click(
      await screen.findByRole("option", { name: "SETTINGS$AGENT_TYPE_ACP" }),
    );
    expect(control().isDirty).toBe(true);

    await user.click(screen.getByTestId("agent-type-selector"));
    await user.click(
      await screen.findByRole("option", {
        name: "SETTINGS$AGENT_TYPE_OPENHANDS",
      }),
    );
    expect(control().isDirty).toBe(false);
  });

  it("saves typed ACP credentials through the save control", async () => {
    const user = userEvent.setup();
    const createSecret = vi.spyOn(SecretsService, "createSecret");
    const { control } = renderAgentSettingsScreen({
      agentSettingsOverride: { agent_kind: "openhands" },
    });
    await screen.findByTestId("agent-settings-screen");

    await user.click(screen.getByTestId("agent-type-selector"));
    await user.click(
      await screen.findByRole("option", { name: "SETTINGS$AGENT_TYPE_ACP" }),
    );
    await user.type(
      await screen.findByTestId("settings-acp-secret-ANTHROPIC_API_KEY"),
      "sk-ant-xyz",
    );
    expect(control().credentials.isDirty).toBe(true);

    let saved = false;
    await act(async () => {
      saved = await control().credentials.save({ silent: true });
    });

    expect(saved).toBe(true);
    expect(createSecret).toHaveBeenCalledWith(
      "ANTHROPIC_API_KEY",
      "sk-ant-xyz",
      undefined,
    );
    // The editor owns the single "saved" toast.
    expect(toastMocks.success).not.toHaveBeenCalled();
  });

  it("a credentials-only edit marks the form dirty", async () => {
    const user = userEvent.setup();
    const { control } = renderAgentSettingsScreen({
      agentSettingsOverride: CLAUDE_PROFILE,
    });
    await screen.findByTestId("agent-command-input");
    expect(control().isDirty).toBe(false);

    await user.type(
      await screen.findByTestId("settings-acp-secret-ANTHROPIC_API_KEY"),
      "sk-ant-only",
    );

    expect(control().isDirty).toBe(true);
  });

  it("shows the 'already signed in' banner in the credentials section when authenticated", async () => {
    acpAuthStatusMock.mockReturnValue({
      status: "authenticated",
      isChecking: false,
      isSupported: true,
    });

    renderAgentSettingsScreen({ agentSettingsOverride: CLAUDE_PROFILE });
    await screen.findByTestId("agent-settings-screen");

    expect(
      await screen.findByTestId("settings-acp-auth-detected"),
    ).toBeInTheDocument();
  });
});

describe("AgentSettingsScreen — MCP server scope", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  function seedWithMcp(mcpConfig: Record<string, unknown>) {
    // `useSettings` prefers `agent_settings.mcp_config` over the top-level
    // field, so seed it where the hook actually reads.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          agent_kind: "openhands",
          mcp_config: mcpConfig,
        },
      } as never),
    );
  }

  const TWO_SERVERS = {
    github: { url: "https://mcp.example/github", transport: "shttp" },
    postgres: { url: "https://mcp.example/pg", transport: "shttp" },
  };

  it("lists configured servers read-only and persists null by default", async () => {
    seedWithMcp(TWO_SERVERS);
    let control: AgentSettingsSaveControl | null = null;
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "openhands",
        enable_sub_agents: false,
        mcp_server_refs: null,
      },
      onSaveControlChange: (next) => {
        control = next;
      },
    });
    await screen.findByTestId("agent-settings-screen");

    const github = screen.getByTestId("agent-settings-mcp-github");
    expect(github).toBeChecked();
    expect(github).toBeDisabled();
    expect(control!.buildAgentProfileFields()).toMatchObject({
      mcp_server_refs: null,
    });
  });

  it("seeds from a stored scope and persists the selection", async () => {
    seedWithMcp(TWO_SERVERS);
    let control: AgentSettingsSaveControl | null = null;
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "openhands",
        enable_sub_agents: false,
        mcp_server_refs: ["github"],
      },
      onSaveControlChange: (next) => {
        control = next;
      },
    });
    await screen.findByTestId("agent-settings-screen");

    expect(screen.getByTestId("agent-settings-mcp-github")).toBeChecked();
    expect(screen.getByTestId("agent-settings-mcp-postgres")).not.toBeChecked();
    expect(control!.buildAgentProfileFields()).toMatchObject({
      mcp_server_refs: ["github"],
    });
  });

  it("seeds a switch to custom with every configured server", async () => {
    // Turning the control on should narrow from the default rather than cut
    // the agent off from every server at once.
    seedWithMcp(TWO_SERVERS);
    let control: AgentSettingsSaveControl | null = null;
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "openhands",
        enable_sub_agents: false,
        mcp_server_refs: null,
      },
      onSaveControlChange: (next) => {
        control = next;
      },
    });
    await screen.findByTestId("agent-settings-screen");

    const user = userEvent.setup();
    const combo = screen.getByTestId("agent-settings-mcp-mode");
    combo.focus();
    await user.keyboard("{ArrowDown}");
    await user.click(
      await screen.findByRole("option", {
        name: "SETTINGS$AGENT_PROFILE_MCP_CHOOSE",
      }),
    );

    await waitFor(() => {
      expect(control!.buildAgentProfileFields()).toMatchObject({
        mcp_server_refs: ["github", "postgres"],
      });
    });
    expect(screen.getByTestId("agent-settings-mcp-github")).toBeChecked();
    expect(screen.getByTestId("agent-settings-mcp-postgres")).toBeChecked();
  });

  it("warns about a ref whose server is gone, which would fail the launch", async () => {
    seedWithMcp(TWO_SERVERS);
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "openhands",
        enable_sub_agents: false,
        mcp_server_refs: ["github", "deleted-server"],
      },
    });
    await screen.findByTestId("agent-settings-screen");

    expect(
      screen.getByTestId("agent-settings-mcp-deleted-server"),
    ).toBeChecked();
    expect(
      screen.getByTestId("agent-settings-mcp-dangling"),
    ).toBeInTheDocument();
  });

  it("scopes an ACP profile too, since the field lives on the profile base", async () => {
    seedWithMcp(TWO_SERVERS);
    let control: AgentSettingsSaveControl | null = null;
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "acp",
        acp_server: "claude-code",
        mcp_server_refs: ["github"],
      },
      onSaveControlChange: (next) => {
        control = next;
      },
    });
    await screen.findByTestId("agent-settings-screen");

    expect(screen.getByTestId("agent-settings-mcp-github")).toBeChecked();
    expect(control!.buildAgentProfileFields()).toMatchObject({
      agent_kind: "acp",
      mcp_server_refs: ["github"],
    });
  });

  it("explains the empty state when no server is configured", async () => {
    seedWithMcp({});
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "openhands",
        enable_sub_agents: false,
      },
    });
    await screen.findByTestId("agent-settings-screen");
    expect(
      screen.queryByTestId("agent-settings-mcp-list"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByText("SETTINGS$AGENT_PROFILE_MCP_NONE"),
    ).toBeInTheDocument();
  });
});

describe("AgentSettingsScreen — MCP scope dirty tracking", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          agent_kind: "openhands",
          mcp_config: {
            github: { url: "https://mcp.example/github", transport: "shttp" },
            postgres: { url: "https://mcp.example/pg", transport: "shttp" },
          },
        },
      } as never),
    );
  });

  async function dirtyForStoredRefs(refs: string[]) {
    let control: AgentSettingsSaveControl | null = null;
    renderAgentSettingsScreen({
      agentSettingsOverride: {
        agent_kind: "openhands",
        enable_sub_agents: false,
        mcp_server_refs: refs,
      },
      onSaveControlChange: (next) => {
        control = next;
      },
    });
    await screen.findByTestId("agent-settings-screen");
    await screen.findByTestId("agent-settings-mcp-list");
    return () => control!.isDirty;
  }

  it("is clean on load when the stored order matches the config order", async () => {
    const isDirty = await dirtyForStoredRefs(["github", "postgres"]);
    await waitFor(() => expect(isDirty()).toBe(false));
  });

  it("is clean on load when the stored order differs from the config order", async () => {
    const isDirty = await dirtyForStoredRefs(["postgres", "github"]);
    await waitFor(() => expect(isDirty()).toBe(false));
  });
  describe("secret scope", () => {
    beforeEach(() => {
      acpAuthStatusMock.mockReturnValue({
        status: "unknown",
        isChecking: false,
        isSupported: true,
      });
      profileSupportsSecretRefsMock.mockReturnValue(true);
      savedSecretsMock.mockReturnValue([
        { name: "GITHUB_TOKEN", description: "repo access" },
        { name: "DATADOG_API_KEY" },
        { name: "PROD_DB_URL" },
      ]);
    });
    function seedOpenHandsSettings() {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
        buildSettings({
          agent_settings: {
            ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
            agent_kind: "openhands",
          },
        }),
      );
    }

    it("lists every saved secret read-only and persists null by default", async () => {
      seedOpenHandsSettings();
      let control: AgentSettingsSaveControl | null = null;
      renderAgentSettingsScreen({
        agentSettingsOverride: {
          agent_kind: "openhands",
          enable_sub_agents: false,
          secret_refs: null,
        },
        onSaveControlChange: (next) => {
          control = next;
        },
      });
      await screen.findByTestId("agent-settings-screen");

      expect(
        screen.getByTestId("agent-settings-secret-list"),
      ).toBeInTheDocument();
      const github = screen.getByTestId("agent-settings-secret-GITHUB_TOKEN");
      expect(github).toBeChecked();
      expect(github).toBeDisabled();
      expect(
        screen.getByTestId("agent-settings-secret-PROD_DB_URL"),
      ).toBeChecked();
      expect(control!.buildAgentProfileFields()).toMatchObject({
        secret_refs: null,
      });
    });

    it("seeds from a stored scope and persists the selection", async () => {
      seedOpenHandsSettings();
      let control: AgentSettingsSaveControl | null = null;
      renderAgentSettingsScreen({
        agentSettingsOverride: {
          agent_kind: "openhands",
          enable_sub_agents: false,
          secret_refs: ["DATADOG_API_KEY"],
        },
        onSaveControlChange: (next) => {
          control = next;
        },
      });
      await screen.findByTestId("agent-settings-screen");

      expect(
        screen.getByTestId("agent-settings-secret-DATADOG_API_KEY"),
      ).toBeChecked();
      expect(
        screen.getByTestId("agent-settings-secret-PROD_DB_URL"),
      ).not.toBeChecked();
      expect(control!.buildAgentProfileFields()).toMatchObject({
        secret_refs: ["DATADOG_API_KEY"],
      });
    });

    it("keeps a stored ref whose secret no longer exists", async () => {
      // The save is a whole-profile overwrite, so dropping it here would
      // silently rewrite the user's scope.
      seedOpenHandsSettings();
      let control: AgentSettingsSaveControl | null = null;
      renderAgentSettingsScreen({
        agentSettingsOverride: {
          agent_kind: "openhands",
          enable_sub_agents: false,
          secret_refs: ["DELETED_SECRET"],
        },
        onSaveControlChange: (next) => {
          control = next;
        },
      });
      await screen.findByTestId("agent-settings-screen");

      expect(
        screen.getByTestId("agent-settings-secret-DELETED_SECRET"),
      ).toBeChecked();
      expect(control!.buildAgentProfileFields()).toMatchObject({
        secret_refs: ["DELETED_SECRET"],
      });
    });

    it.each([{ secretRefs: [] }, { secretRefs: ["PROD_DB_URL"] }])(
      "preserves an existing ACP secret scope $secretRefs through an unrelated edit",
      async ({ secretRefs }) => {
        savedSecretsMock.mockReturnValue([
          { name: "ANTHROPIC_API_KEY" },
          { name: "PROD_DB_URL" },
        ]);
        seedOpenHandsSettings();
        let control: AgentSettingsSaveControl | null = null;
        renderAgentSettingsScreen({
          agentSettingsOverride: {
            agent_kind: "acp",
            acp_server: "claude-code",
            acp_command: [...CLAUDE_COMMAND],
            acp_args: [],
            acp_model: "haiku",
            secret_refs: secretRefs,
          },
          onSaveControlChange: (next) => {
            control = next;
          },
        });
        await screen.findByTestId("agent-command-input");
        expect(control!.isDirty).toBe(false);
        expect(
          screen.getByTestId("agent-settings-secret-ANTHROPIC_API_KEY"),
        ).not.toBeChecked();

        const user = userEvent.setup();
        await user.click(screen.getByTestId("agent-settings-mcp-mode"));
        await user.click(
          await screen.findByRole("option", {
            name: "SETTINGS$AGENT_PROFILE_MCP_CHOOSE",
          }),
        );
        expect(control!.buildAgentProfileFields()).toMatchObject({
          mcp_server_refs: [],
          secret_refs: secretRefs,
        });
      },
    );

    it("keeps a provider credential deselected when the saved ACP profile is reopened", async () => {
      savedSecretsMock.mockReturnValue([
        { name: "ANTHROPIC_API_KEY" },
        { name: "PROD_DB_URL" },
      ]);
      seedOpenHandsSettings();
      let control: AgentSettingsSaveControl | null = null;
      const onSaveControlChange = (next: AgentSettingsSaveControl | null) => {
        control = next;
      };
      const view = renderAgentSettingsScreen({
        agentSettingsOverride: {
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: [...CLAUDE_COMMAND],
          acp_args: [],
          acp_model: "haiku",
          secret_refs: ["ANTHROPIC_API_KEY", "PROD_DB_URL"],
        },
        onSaveControlChange,
      });
      await screen.findByTestId("agent-command-input");
      const user = userEvent.setup();
      await user.click(
        screen.getByTestId("agent-settings-secret-ANTHROPIC_API_KEY"),
      );
      const saved = control!.buildAgentProfileFields();
      expect(saved).toMatchObject({ secret_refs: ["PROD_DB_URL"] });
      view.unmount();
      renderAgentSettingsScreen({
        agentSettingsOverride: saved,
        onSaveControlChange,
      });
      await screen.findByTestId("agent-command-input");
      expect(
        screen.getByTestId("agent-settings-secret-ANTHROPIC_API_KEY"),
      ).not.toBeChecked();
      expect(control!.isDirty).toBe(false);
      expect(control!.buildAgentProfileFields()).toMatchObject({
        secret_refs: ["PROD_DB_URL"],
      });
    });

    it.each([true, false])(
      "selects provider credentials before saving (already stored: %s)",
      async (alreadyStored) => {
        // Scoping is strict server-side, so an ACP profile that omits its
        // credential cannot authenticate. Seed it visibly rather than re-adding
        // it behind the user's back.
        savedSecretsMock.mockReturnValue([
          ...(alreadyStored
            ? [{ name: "ANTHROPIC_API_KEY" }, { name: "ANTHROPIC_BASE_URL" }]
            : []),
          { name: "PROD_DB_URL" },
        ]);
        seedOpenHandsSettings();
        let control: AgentSettingsSaveControl | null = null;
        renderAgentSettingsScreen({
          agentSettingsOverride: {
            agent_kind: "acp",
            acp_server: "claude-code",
            // From the registry, not a literal: the pinned command carries a
            // version that moves, and a stale one detects as `custom` (no
            // provider credentials) instead of failing loudly.
            acp_command: [...CLAUDE_COMMAND],
            acp_args: [],
            acp_model: "",
          },
          onSaveControlChange: (next) => {
            control = next;
          },
        });
        await screen.findByTestId("agent-settings-screen");

        const user = userEvent.setup();
        await user.click(screen.getByTestId("agent-settings-secrets-mode"));
        await user.click(
          await screen.findByRole("option", {
            name: "SETTINGS$AGENT_PROFILE_SECRETS_CHOOSE",
          }),
        );

        await waitFor(() => {
          expect(
            screen.getByTestId("agent-settings-secret-ANTHROPIC_API_KEY"),
          ).toBeChecked();
        });
        // Seeded, not forced: an unrelated secret stays off.
        expect(
          screen.getByTestId("agent-settings-secret-PROD_DB_URL"),
        ).not.toBeChecked();

        await user.click(
          screen.getByTestId("agent-settings-secret-ANTHROPIC_API_KEY"),
        );
        expect(
          screen.getByTestId("agent-settings-secret-ANTHROPIC_API_KEY"),
        ).not.toBeChecked();
        expect(control!.buildAgentProfileFields()).toMatchObject({
          secret_refs: expect.not.arrayContaining(["ANTHROPIC_API_KEY"]),
        });
        await user.click(
          screen.getByTestId("agent-settings-secret-ANTHROPIC_API_KEY"),
        );
        const refs = (
          control!.buildAgentProfileFields() as { secret_refs?: string[] }
        ).secret_refs;
        expect(refs).toContain("ANTHROPIC_API_KEY");
      },
    );

    it.each(["preset", "command"])(
      "selects the new provider credential after an explicit %s change",
      async (input) => {
        savedSecretsMock.mockReturnValue([
          { name: "ANTHROPIC_API_KEY" },
          { name: "OPENAI_API_KEY" },
        ]);
        seedOpenHandsSettings();
        renderAgentSettingsScreen({
          agentSettingsOverride: {
            agent_kind: "acp",
            acp_server: "claude-code",
            acp_command: [...CLAUDE_COMMAND],
            acp_args: [],
            acp_model: "haiku",
            secret_refs: [],
          },
        });
        await screen.findByTestId("agent-command-input");
        const user = userEvent.setup();
        const codex = ACP_PROVIDERS.find(
          (provider) => provider.key === "codex",
        )!;
        if (input === "preset") {
          await user.click(screen.getByTestId("agent-preset-selector"));
          await user.click(
            await screen.findByRole("option", { name: codex.display_name }),
          );
        } else {
          const command = screen.getByTestId("agent-command-input");
          await user.clear(command);
          await user.type(command, codex.default_command.join(" "));
        }
        expect(
          screen.getByTestId("agent-settings-secret-OPENAI_API_KEY"),
        ).toBeChecked();
        expect(
          screen.getByTestId("agent-settings-secret-ANTHROPIC_API_KEY"),
        ).not.toBeChecked();
      },
    );

    it("leaves an OpenHands profile's scope empty when scoping starts", async () => {
      // Nothing an OpenHands agent needs rides this channel, so there is
      // nothing to seed.
      seedOpenHandsSettings();
      let control: AgentSettingsSaveControl | null = null;
      renderAgentSettingsScreen({
        agentSettingsOverride: {
          agent_kind: "openhands",
          enable_sub_agents: false,
        },
        onSaveControlChange: (next) => {
          control = next;
        },
      });
      await screen.findByTestId("agent-settings-screen");

      const user = userEvent.setup();
      await user.click(screen.getByTestId("agent-settings-secrets-mode"));
      await user.click(
        await screen.findByRole("option", {
          name: "SETTINGS$AGENT_PROFILE_SECRETS_CHOOSE",
        }),
      );

      await waitFor(() => {
        expect(control!.buildAgentProfileFields()).toMatchObject({
          secret_refs: [],
        });
      });
    });

    it("omits the key on a backend whose profile model predates it", async () => {
      profileSupportsSecretRefsMock.mockReturnValue(false);
      seedOpenHandsSettings();
      let control: AgentSettingsSaveControl | null = null;
      renderAgentSettingsScreen({
        agentSettingsOverride: {
          agent_kind: "openhands",
          enable_sub_agents: false,
        },
        onSaveControlChange: (next) => {
          control = next;
        },
      });
      await screen.findByTestId("agent-settings-screen");

      expect(
        screen.queryByTestId("agent-settings-secrets-mode"),
      ).not.toBeInTheDocument();
      expect(control!.buildAgentProfileFields()).not.toHaveProperty(
        "secret_refs",
      );
    });

    it("explains the empty state when nothing is saved", async () => {
      savedSecretsMock.mockReturnValue([]);
      seedOpenHandsSettings();
      renderAgentSettingsScreen({
        agentSettingsOverride: {
          agent_kind: "openhands",
          enable_sub_agents: false,
        },
      });
      await screen.findByTestId("agent-settings-screen");
      expect(
        screen.queryByTestId("agent-settings-secret-list"),
      ).not.toBeInTheDocument();
      expect(
        screen.getByText("SETTINGS$AGENT_PROFILE_SECRETS_NONE"),
      ).toBeInTheDocument();
      // The control itself stays, so the user can still see the scope mode.
      expect(
        screen.getByTestId("agent-settings-secrets-mode"),
      ).toBeInTheDocument();
    });
  });
});

describe("AgentSettingsScreen — system prompt", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([]);
    profileSupportsInstructionsMock.mockReturnValue(true);
    profileSupportsSystemPromptMock.mockReturnValue(true);
  });

  const OPENHANDS_PROFILE = {
    agent_kind: "openhands",
    enable_sub_agents: false,
    mcp_server_refs: null,
  };

  async function openModes() {
    const user = userEvent.setup();
    const combo = screen.getByTestId("agent-settings-system-prompt-mode");
    combo.focus();
    await user.keyboard("{ArrowDown}");
    return user;
  }

  async function chooseMode(name: string) {
    const user = await openModes();
    await user.click(await screen.findByRole("option", { name }));
    return user;
  }

  it("is hidden where launches would not apply it", async () => {
    profileSupportsInstructionsMock.mockReturnValue(false);
    const { control } = renderAgentSettingsScreen({
      agentSettingsOverride: OPENHANDS_PROFILE,
    });
    await screen.findByTestId("agent-settings-screen");

    expect(
      screen.queryByTestId("agent-settings-system-prompt-mode"),
    ).not.toBeInTheDocument();
    const fields = control().buildAgentProfileFields();
    expect(fields).not.toHaveProperty("system_prompt");
    expect(fields).not.toHaveProperty("system_message_suffix");
  });

  it("offers a custom prompt only when the profile model accepts it", async () => {
    profileSupportsSystemPromptMock.mockReturnValue(false);
    renderAgentSettingsScreen({ agentSettingsOverride: OPENHANDS_PROFILE });
    await screen.findByTestId("agent-settings-system-prompt-mode");

    await openModes();
    expect(
      await screen.findByRole("option", {
        name: "SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT_APPEND",
      }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("option", {
        name: "SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT_CUSTOM",
      }),
    ).not.toBeInTheDocument();
  });

  it("persists null for both fields on the OpenHands default", async () => {
    const { control } = renderAgentSettingsScreen({
      agentSettingsOverride: OPENHANDS_PROFILE,
    });
    await screen.findByTestId("agent-settings-system-prompt-mode");

    expect(
      screen.queryByTestId("agent-settings-system-prompt"),
    ).not.toBeInTheDocument();
    expect(control().buildAgentProfileFields()).toMatchObject({
      system_prompt: null,
      system_message_suffix: null,
    });
  });

  it.each([
    ["system_message_suffix", "append"],
    ["system_prompt", "custom"],
  ])("opens a stored %s in %s mode, clean", async (field, mode) => {
    const { control } = renderAgentSettingsScreen({
      agentSettingsOverride: { ...OPENHANDS_PROFILE, [field]: "Stored text." },
    });

    expect(
      await screen.findByTestId("agent-settings-system-prompt"),
    ).toHaveValue("Stored text.");
    expect(
      screen.getByTestId("agent-settings-system-prompt-hint"),
    ).toHaveTextContent(
      mode === "append"
        ? "SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT_APPEND_HINT"
        : "SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT_CUSTOM_HINT",
    );
    expect(control().isDirty).toBe(false);
    expect(control().buildAgentProfileFields()).toMatchObject({
      [field]: "Stored text.",
    });
  });

  it("saves typed instructions as the suffix, not an override", async () => {
    const { control } = renderAgentSettingsScreen({
      agentSettingsOverride: OPENHANDS_PROFILE,
    });
    await screen.findByTestId("agent-settings-system-prompt-mode");

    const user = await chooseMode(
      "SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT_APPEND",
    );
    await user.type(
      screen.getByTestId("agent-settings-system-prompt"),
      "Be terse.",
    );
    await waitFor(() => expect(control().isDirty).toBe(true));
    expect(control().buildAgentProfileFields()).toMatchObject({
      system_prompt: null,
      system_message_suffix: "Be terse.",
    });
  });

  it("blocks saving empty text, then saves the typed custom prompt", async () => {
    const { control } = renderAgentSettingsScreen({
      agentSettingsOverride: OPENHANDS_PROFILE,
    });
    await screen.findByTestId("agent-settings-system-prompt-mode");

    const user = await chooseMode(
      "SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT_CUSTOM",
    );
    await waitFor(() => expect(control().isValid).toBe(false));
    expect(
      screen.getByTestId("agent-settings-system-prompt-required"),
    ).toBeInTheDocument();

    await user.type(
      screen.getByTestId("agent-settings-system-prompt"),
      "You review PRs.",
    );
    await waitFor(() => expect(control().isValid).toBe(true));
    expect(control().buildAgentProfileFields()).toMatchObject({
      system_prompt: "You review PRs.",
      system_message_suffix: null,
    });
  });

  it("clears a stored prompt when switched back to the default", async () => {
    const { control } = renderAgentSettingsScreen({
      agentSettingsOverride: {
        ...OPENHANDS_PROFILE,
        system_prompt: "You triage issues.",
      },
    });
    await screen.findByTestId("agent-settings-system-prompt");

    await chooseMode("SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT_STANDARD");
    await waitFor(() => expect(control().isDirty).toBe(true));
    expect(control().buildAgentProfileFields()).toMatchObject({
      system_prompt: null,
      system_message_suffix: null,
    });
  });

  it("explains instead of editing on the default profile, and leaves both fields alone", async () => {
    const { control } = renderAgentSettingsScreen({
      agentSettingsOverride: {
        ...OPENHANDS_PROFILE,
        system_prompt: "Set through the API.",
        system_message_suffix: "Also set through the API.",
      },
      isDefaultProfile: true,
    });

    expect(
      await screen.findByTestId("agent-settings-system-prompt-default-profile"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("agent-settings-system-prompt-mode"),
    ).not.toBeInTheDocument();
    const fields = control().buildAgentProfileFields();
    expect(fields).not.toHaveProperty("system_prompt");
    expect(fields).not.toHaveProperty("system_message_suffix");
  });
});
