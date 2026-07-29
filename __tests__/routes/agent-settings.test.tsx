import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AgentSettingsScreen } from "#/routes/agent-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import { SecretsService } from "#/api/secrets-service";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { Settings } from "#/types/settings";

// Stub the login-detection probe so the ACP credentials section doesn't spin a
// subprocess; default to no detected session so existing tests are unaffected.
const acpAuthStatusMock = vi.hoisted(() => vi.fn());
vi.mock("#/hooks/query/use-acp-auth-status", () => ({
  useAcpAuthStatus: (...args: unknown[]) => acpAuthStatusMock(...args),
}));

// Observe save toasts so we can assert the single Save shows one confirmation,
// not one per persisted thing (agent spec + credentials).
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

function renderAgentSettingsScreen() {
  return render(<AgentSettingsScreen />, {
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

describe("AgentSettingsScreen", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);
    // The page owns the ACP credential form (single Save), so it reads/writes
    // secrets even on non-ACP renders.
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
  });

  it("renders the agent type selector defaulting to OpenHands with sub-agents toggle", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          agent_kind: "openhands",
        },
      }),
    );

    renderAgentSettingsScreen();
    await screen.findByTestId("agent-settings-screen");
    expect(screen.getByTestId("agent-type-selector")).toBeInTheDocument();
    // Sub-agents toggle visible on the OpenHands branch.
    expect(
      screen.getByTestId("agent-settings-enable-sub-agents"),
    ).toBeInTheDocument();
    // ACP-only fields stay hidden on the OpenHands branch.
    expect(screen.queryByTestId("agent-command-input")).not.toBeInTheDocument();
  });

  it("labels the save button 'Save Changes' for consistency with other settings pages", async () => {
    // Arrange — render with any valid settings; the label is independent
    // of the form's state.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          agent_kind: "openhands",
        },
      }),
    );

    // Act
    renderAgentSettingsScreen();
    await screen.findByTestId("agent-settings-screen");

    // Assert — t() is stubbed to return the key, so the rendered text is
    // the translation key. SETTINGS$SAVE_CHANGES = "Save Changes" in
    // public/locales/en/openhands.json; BUTTON$SAVE = "Save" (the bug).
    expect(screen.getByTestId("agent-save-button")).toHaveTextContent(
      "SETTINGS$SAVE_CHANGES",
    );
  });

  it("saves enable_sub_agents when toggling on the OpenHands path", async () => {
    const user = userEvent.setup();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          agent_kind: "openhands",
          enable_sub_agents: false,
        },
      }),
    );
    const save = vi.spyOn(SettingsService, "saveSettings");

    renderAgentSettingsScreen();
    await screen.findByTestId("agent-settings-screen");

    // Toggle sub-agents on via the enclosing label
    const toggle = screen.getByTestId("agent-settings-enable-sub-agents");
    const label = toggle.closest("label")!;
    await user.click(label);

    await user.click(screen.getByTestId("agent-save-button"));

    await waitFor(() => {
      expect(save).toHaveBeenCalledTimes(1);
    });
    const call = save.mock.calls[0]?.[0] as {
      agent_settings_diff?: Record<string, unknown>;
    };
    expect(call.agent_settings_diff).toEqual({
      agent_kind: "openhands",
      enable_sub_agents: true,
      tool_concurrency_limit: 1,
    });
  });

  it("saves tool_concurrency_limit when changed on the OpenHands path", async () => {
    const user = userEvent.setup();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          agent_kind: "openhands",
          enable_sub_agents: false,
          tool_concurrency_limit: 1,
        },
      }),
    );
    const save = vi.spyOn(SettingsService, "saveSettings");

    renderAgentSettingsScreen();
    await screen.findByTestId("agent-settings-screen");

    const input = screen.getByTestId("sdk-settings-tool_concurrency_limit");
    await user.clear(input);
    await user.type(input, "4");

    await user.click(screen.getByTestId("agent-save-button"));

    await waitFor(() => {
      expect(save).toHaveBeenCalledTimes(1);
    });
    const call = save.mock.calls[0]?.[0] as {
      agent_settings_diff?: Record<string, unknown>;
    };
    // Coerced to a number (not the raw input string) via the shared
    // schema-driven coercion.
    expect(call.agent_settings_diff?.tool_concurrency_limit).toBe(4);
  });

  it("hides sub-agents toggle when ACP is selected", async () => {
    const user = userEvent.setup();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          agent_kind: "openhands",
        },
      }),
    );

    renderAgentSettingsScreen();
    await screen.findByTestId("agent-settings-screen");

    // Sub-agents toggle should be visible initially
    expect(
      screen.getByTestId("agent-settings-enable-sub-agents"),
    ).toBeInTheDocument();

    // Switch to ACP
    await user.click(screen.getByTestId("agent-type-selector"));
    await user.click(
      await screen.findByRole("option", { name: "SETTINGS$AGENT_TYPE_ACP" }),
    );

    // Sub-agents toggle should be hidden, ACP fields should appear
    expect(
      screen.queryByTestId("agent-settings-enable-sub-agents"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("agent-command-input")).toBeInTheDocument();
  });

  it("shows the ACP form when the active agent_kind is acp", async () => {
    // Use a model ID that isn't in CLAUDE_MODELS so the form falls through to
    // the custom-input branch — that's the path this test is asserting (saved
    // value round-trips into the visible input). Known IDs go through the
    // dropdown instead and are covered separately.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: ["npx", "-y", "@agentclientprotocol/claude-agent-acp"],
          acp_model: "my-pinned-fork-model",
        },
      }),
    );

    renderAgentSettingsScreen();
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
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: [],
          acp_model: null,
        },
      }),
    );

    renderAgentSettingsScreen();

    await screen.findByTestId("agent-command-input");
    expect(screen.getByLabelText("SETTINGS$AGENT_MODEL")).toHaveValue(
      "Claude Opus 4.8 (1M)",
    );
  });

  it("saves the selected built-in ACP model", async () => {
    const user = userEvent.setup();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: [],
          acp_model: null,
        },
      }),
    );
    const save = vi.spyOn(SettingsService, "saveSettings");

    renderAgentSettingsScreen();
    await screen.findByTestId("agent-command-input");
    await user.click(screen.getByLabelText("SETTINGS$AGENT_MODEL"));
    await user.click(await screen.findByText("Claude Haiku 4.5"));
    await user.click(screen.getByTestId("agent-save-button"));

    await waitFor(() => {
      expect(save).toHaveBeenCalledTimes(1);
    });
    const call = save.mock.calls[0]?.[0] as {
      agent_settings_diff?: Record<string, unknown>;
    };
    expect(call.agent_settings_diff?.acp_model).toBe("haiku");
  });

  it("clears the model when switching from a built-in provider to Custom", async () => {
    // F3 from review: built-ins seed ``acp_model`` to their registered
    // ``default_model`` on load. Picking Custom must not leak that built-in
    // default into custom settings — otherwise a user choosing Custom from
    // Claude Code would silently save ``acp_model: "claude-opus-4-7"`` on an
    // unrelated wrapper.
    const user = userEvent.setup();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: [],
          acp_model: null,
        },
      }),
    );
    const save = vi.spyOn(SettingsService, "saveSettings");

    renderAgentSettingsScreen();
    await screen.findByTestId("agent-command-input");
    // Form loads with the Claude Code default visible.
    expect(screen.getByLabelText("SETTINGS$AGENT_MODEL")).toHaveValue(
      "Claude Opus 4.8 (1M)",
    );

    // Switch to the Custom preset, then enter a different command — the
    // form's ``selectedPreset`` re-derives from the command text, so the
    // save path only treats it as Custom once the command no longer matches
    // a built-in provider's default.
    await user.click(screen.getByTestId("agent-preset-selector"));
    await user.click(
      await screen.findByRole("option", {
        name: "SETTINGS$AGENT_PRESET_CUSTOM",
      }),
    );
    const commandInput = screen.getByTestId(
      "agent-command-input",
    ) as HTMLTextAreaElement;
    await user.clear(commandInput);
    await user.type(commandInput, "my-custom-acp --flag");

    await user.click(screen.getByTestId("agent-save-button"));
    await waitFor(() => {
      expect(save).toHaveBeenCalledTimes(1);
    });

    const call = save.mock.calls[0]?.[0] as {
      agent_settings_diff?: Record<string, unknown>;
    };
    // Custom preset has no registered default — saved acp_model must be null,
    // not the inherited Claude Opus default.
    expect(call.agent_settings_diff?.acp_server).toBe("custom");
    expect(call.agent_settings_diff?.acp_model).toBeNull();
  });

  it("reconciles the model when the command is retyped to a different provider", async () => {
    // Editing the command textarea (rather than the preset dropdown) into a
    // different built-in provider must not leave the previous provider's model
    // selected — otherwise Save would persist e.g. ``claude-opus-4-7`` against
    // a Codex wrapper. The detected preset changes, so the model reconciles to
    // the new provider's default.
    const user = userEvent.setup();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: [],
          acp_model: null,
        },
      }),
    );
    const save = vi.spyOn(SettingsService, "saveSettings");

    renderAgentSettingsScreen();
    await screen.findByTestId("agent-command-input");
    expect(screen.getByLabelText("SETTINGS$AGENT_MODEL")).toHaveValue(
      "Claude Opus 4.8 (1M)",
    );

    const commandInput = screen.getByTestId(
      "agent-command-input",
    ) as HTMLTextAreaElement;
    await user.clear(commandInput);
    await user.type(
      commandInput,
      "npx -y @agentclientprotocol/codex-acp@1.1.2",
    );

    // The model field now reflects the Codex default, not the stale Claude one.
    expect(screen.getByLabelText("SETTINGS$AGENT_MODEL")).toHaveValue(
      "GPT-5.5",
    );

    await user.click(screen.getByTestId("agent-save-button"));
    await waitFor(() => {
      expect(save).toHaveBeenCalledTimes(1);
    });

    const call = save.mock.calls[0]?.[0] as {
      agent_settings_diff?: Record<string, unknown>;
    };
    expect(call.agent_settings_diff?.acp_server).toBe("codex");
    expect(call.agent_settings_diff?.acp_model).toBe("gpt-5.5");
  });

  it("saves an ACP diff when switching to ACP + Claude Code", async () => {
    const user = userEvent.setup();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          agent_kind: "openhands",
        },
      }),
    );
    const save = vi.spyOn(SettingsService, "saveSettings");

    renderAgentSettingsScreen();
    await screen.findByTestId("agent-settings-screen");

    // Switching to ACP prefills the command from the first registered provider
    // (Claude Code).
    await user.click(screen.getByTestId("agent-type-selector"));
    await user.click(
      await screen.findByRole("option", { name: "SETTINGS$AGENT_TYPE_ACP" }),
    );

    const commandInput = (await screen.findByTestId(
      "agent-command-input",
    )) as HTMLTextAreaElement;
    expect(commandInput.value).toBe(
      "npx -y @agentclientprotocol/claude-agent-acp@0.44.0",
    );
    expect(screen.getByLabelText("SETTINGS$AGENT_MODEL")).toHaveValue(
      "Claude Opus 4.8 (1M)",
    );

    await user.click(screen.getByTestId("agent-save-button"));

    await waitFor(() => {
      expect(save).toHaveBeenCalledTimes(1);
    });
    const call = save.mock.calls[0]?.[0] as {
      agent_settings_diff?: Record<string, unknown>;
    };
    expect(call.agent_settings_diff).toEqual({
      agent_kind: "acp",
      acp_server: "claude-code",
      // The default-command path stores acp_command: [] and lets the registry
      // resolve it on the agent-server side. Round-tripping verbatim would
      // pin a stale command if the registry default changes upstream.
      acp_command: [],
      // ``acp_args: []`` is reset on every save so an API-set
      // ``acp_args`` can't survive and concatenate onto the spawn
      // command at conversation-create time.
      acp_args: [],
      acp_model: "opus[1m]",
    });
  });

  it("clears ACP fields when switching back to OpenHands", async () => {
    const user = userEvent.setup();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: ["npx", "-y", "@agentclientprotocol/claude-agent-acp"],
        },
      }),
    );
    const save = vi.spyOn(SettingsService, "saveSettings");

    renderAgentSettingsScreen();
    await screen.findByTestId("agent-settings-screen");

    await user.click(screen.getByTestId("agent-type-selector"));
    await user.click(
      await screen.findByRole("option", {
        name: "SETTINGS$AGENT_TYPE_OPENHANDS",
      }),
    );
    await user.click(screen.getByTestId("agent-save-button"));

    await waitFor(() => {
      expect(save).toHaveBeenCalledTimes(1);
    });
    const call = save.mock.calls[0]?.[0] as {
      agent_settings_diff?: Record<string, unknown>;
    };
    expect(call.agent_settings_diff).toEqual({
      agent_kind: "openhands",
      enable_sub_agents: false,
      tool_concurrency_limit: 1,
    });
  });

  it("disables Save when the user has cleared the command on the ACP path", async () => {
    const user = userEvent.setup();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: ["npx", "-y", "@agentclientprotocol/claude-agent-acp"],
        },
      }),
    );

    renderAgentSettingsScreen();
    const cmd = (await screen.findByTestId(
      "agent-command-input",
    )) as HTMLTextAreaElement;
    const save = screen.getByTestId("agent-save-button") as HTMLButtonElement;

    // Clear the field. Save should be disabled (the agent-server would
    // crash on an empty acp_command and the adapter has no way to
    // recover — better to block the save than silently submit garbage).
    await user.clear(cmd);
    expect(save).toBeDisabled();
  });

  it("treats whitespace-only as empty and keeps Save disabled", async () => {
    const user = userEvent.setup();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: ["npx", "-y", "@agentclientprotocol/claude-agent-acp"],
        },
      }),
    );

    renderAgentSettingsScreen();
    const cmd = (await screen.findByTestId(
      "agent-command-input",
    )) as HTMLTextAreaElement;
    const save = screen.getByTestId("agent-save-button") as HTMLButtonElement;
    await user.clear(cmd);
    await user.type(cmd, "   \t   ");
    expect(save).toBeDisabled();
  });

  it("preserves a Custom command with quoted args end-to-end", async () => {
    // Regression guard for the .split-vs-shell-quote bug: a Custom
    // command like ``bash -c "echo hi"`` used to get tokenised as
    // ``["bash","-c","\"echo","hi\""]`` and silently fail at spawn.
    const user = userEvent.setup();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          agent_kind: "openhands",
        },
      }),
    );
    const save = vi.spyOn(SettingsService, "saveSettings");

    renderAgentSettingsScreen();
    await screen.findByTestId("agent-settings-screen");

    await user.click(screen.getByTestId("agent-type-selector"));
    await user.click(
      await screen.findByRole("option", { name: "SETTINGS$AGENT_TYPE_ACP" }),
    );
    const cmd = (await screen.findByTestId(
      "agent-command-input",
    )) as HTMLTextAreaElement;
    await user.clear(cmd);
    await user.type(cmd, 'bash -c "echo hi"');
    await user.click(screen.getByTestId("agent-save-button"));

    await waitFor(() => {
      expect(save).toHaveBeenCalledTimes(1);
    });
    const call = save.mock.calls[0]?.[0] as {
      agent_settings_diff?: Record<string, unknown>;
    };
    expect(call.agent_settings_diff?.acp_command).toEqual([
      "bash",
      "-c",
      "echo hi",
    ]);
    // Anything that diverges from a built-in default-command snaps to
    // the Custom preset.
    expect(call.agent_settings_diff?.acp_server).toBe("custom");
  });

  it("preserves the registry default when acp_command:[] + non-empty acp_args is loaded", async () => {
    // Regression guard for the data-corruption bug:
    //
    //   stored: acp_server: 'claude-code', acp_command: [], acp_args:
    //           ['--extra-arg']
    //   actual spawn: ['npx', '-y', '@agentclientprotocol/claude-agent-acp',
    //                  '--extra-arg']
    //
    // The form used to merge acp_command + acp_args literally and would
    // show only ``--extra-arg`` in the textarea. Saving then sent
    // ``acp_command: ['--extra-arg']`` and flipped the preset to
    // ``custom``, silently dropping the registry-default prefix.
    // The load path must expand the default *before* merging with args.
    const user = userEvent.setup();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: [],
          acp_args: ["--extra-arg"],
        },
      }),
    );
    const save = vi.spyOn(SettingsService, "saveSettings");

    renderAgentSettingsScreen();
    const cmd = (await screen.findByTestId(
      "agent-command-input",
    )) as HTMLTextAreaElement;
    expect(cmd.value).toBe(
      "npx -y @agentclientprotocol/claude-agent-acp@0.44.0 --extra-arg",
    );

    // Touch the form to mark it dirty (Save is disabled until isDirty),
    // then submit. The data the form sends has to carry the registry-
    // default prefix the user can now SEE in the textarea, not the bare
    // ``--extra-arg`` that was stored.
    await user.click(cmd);
    await user.keyboard("{End} ");
    await user.keyboard("{Backspace}");

    await user.click(screen.getByTestId("agent-save-button"));
    await waitFor(() => {
      expect(save).toHaveBeenCalledTimes(1);
    });
    const call = save.mock.calls[0]?.[0] as {
      agent_settings_diff?: Record<string, unknown>;
    };
    expect(call.agent_settings_diff?.acp_server).toBe("custom");
    expect(call.agent_settings_diff?.acp_command).toEqual([
      "npx",
      "-y",
      "@agentclientprotocol/claude-agent-acp@0.44.0",
      "--extra-arg",
    ]);
    // ``acp_args: []`` resets the API-set args so they don't double up
    // at spawn time.
    expect(call.agent_settings_diff?.acp_args).toEqual([]);
  });

  it("preserves an unknown loaded acp_server when the user saves without editing", async () => {
    // Data-corruption regression: a user with an ``acp_server`` value
    // canvas's registry doesn't know about (e.g. set via the API for a
    // future provider that hasn't been mirrored into ``ACP_PROVIDERS``
    // yet) opens Settings → Agent and clicks Save. Without preservation
    // the save flow demotes ``acp_server: "amp"`` → ``acp_server:
    // "custom"`` because ``detectPreset`` returns ``custom`` for any
    // unknown server. The original key name is silently lost.
    //
    // The fix is narrow: when the user hasn't touched the command since
    // load AND the loaded server is non-empty, non-``"custom"``, and
    // absent from ``ACP_PROVIDERS``, write the loaded key back verbatim
    // via the ``allowUnknownServer`` pass-through.
    const user = userEvent.setup();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "amp",
          acp_command: ["npx", "-y", "@some-future/amp-acp"],
          acp_args: [],
        },
      }),
    );
    const save = vi.spyOn(SettingsService, "saveSettings");

    renderAgentSettingsScreen();
    const cmd = (await screen.findByTestId(
      "agent-command-input",
    )) as HTMLTextAreaElement;
    expect(cmd.value).toBe("npx -y @some-future/amp-acp");

    // Touch + revert the textarea to flip isDirty without changing
    // the persisted command text — matches "user opens settings and
    // hits Save without intending to change anything."
    await user.click(cmd);
    await user.keyboard("{End} ");
    await user.keyboard("{Backspace}");

    await user.click(screen.getByTestId("agent-save-button"));
    await waitFor(() => {
      expect(save).toHaveBeenCalledTimes(1);
    });
    const call = save.mock.calls[0]?.[0] as {
      agent_settings_diff?: Record<string, unknown>;
    };
    expect(call.agent_settings_diff?.acp_server).toBe("amp");
    expect(call.agent_settings_diff?.acp_command).toEqual([
      "npx",
      "-y",
      "@some-future/amp-acp",
    ]);
  });

  it("demotes an unknown loaded acp_server to 'custom' when the user edits the command", async () => {
    // Counterpart to the preserve test: editing the command is a
    // material change of configuration, so it's correct to drop the
    // unknown ``amp`` key and fall back to ``"custom"``. The user is
    // configuring a new command, not preserving the prior one — so
    // the preset name follows the command.
    const user = userEvent.setup();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "amp",
          acp_command: ["npx", "-y", "@some-future/amp-acp"],
          acp_args: [],
        },
      }),
    );
    const save = vi.spyOn(SettingsService, "saveSettings");

    renderAgentSettingsScreen();
    const cmd = (await screen.findByTestId(
      "agent-command-input",
    )) as HTMLTextAreaElement;

    // Actually change the command — append a flag so the textarea
    // differs from the loaded value.
    await user.click(cmd);
    await user.keyboard("{End} --new-flag");

    await user.click(screen.getByTestId("agent-save-button"));
    await waitFor(() => {
      expect(save).toHaveBeenCalledTimes(1);
    });
    const call = save.mock.calls[0]?.[0] as {
      agent_settings_diff?: Record<string, unknown>;
    };
    expect(call.agent_settings_diff?.acp_server).toBe("custom");
    expect(call.agent_settings_diff?.acp_command).toEqual([
      "npx",
      "-y",
      "@some-future/amp-acp",
      "--new-flag",
    ]);
  });

  it("a single Save persists ACP credentials together with the agent spec", async () => {
    const user = userEvent.setup();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          agent_kind: "openhands",
        },
      }),
    );
    const saveSettings = vi.spyOn(SettingsService, "saveSettings");
    const createSecret = vi.spyOn(SecretsService, "createSecret");

    renderAgentSettingsScreen();
    await screen.findByTestId("agent-settings-screen");

    // Switch to ACP (Claude Code prefilled) and paste a credential, then click
    // the single page-level Save button.
    await user.click(screen.getByTestId("agent-type-selector"));
    await user.click(
      await screen.findByRole("option", { name: "SETTINGS$AGENT_TYPE_ACP" }),
    );
    await user.type(
      await screen.findByTestId("settings-acp-secret-ANTHROPIC_API_KEY"),
      "sk-ant-xyz",
    );

    await user.click(screen.getByTestId("agent-save-button"));

    // One click persists both: the credential as a secret AND the agent spec.
    await waitFor(() => {
      expect(createSecret).toHaveBeenCalledWith(
        "ANTHROPIC_API_KEY",
        "sk-ant-xyz",
        undefined,
      );
      expect(saveSettings).toHaveBeenCalledTimes(1);
    });
    const diff = (
      saveSettings.mock.calls[0]?.[0] as {
        agent_settings_diff?: Record<string, unknown>;
      }
    ).agent_settings_diff;
    expect(diff?.agent_kind).toBe("acp");
    expect(diff?.acp_server).toBe("claude-code");

    // One click → one confirmation, even though it persisted both the spec and
    // the credential (the credential save is silenced so it doesn't double up).
    expect(toastMocks.success).toHaveBeenCalledTimes(1);
  });

  it("a credentials-only change saves the secret without re-writing settings", async () => {
    const user = userEvent.setup();
    // Already on ACP, so loading introduces no settings change.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: [],
        },
      }),
    );
    const saveSettings = vi.spyOn(SettingsService, "saveSettings");
    const createSecret = vi.spyOn(SecretsService, "createSecret");

    renderAgentSettingsScreen();
    await screen.findByTestId("agent-settings-screen");

    await user.type(
      await screen.findByTestId("settings-acp-secret-ANTHROPIC_API_KEY"),
      "sk-ant-only",
    );
    await user.click(screen.getByTestId("agent-save-button"));

    await waitFor(() => {
      expect(createSecret).toHaveBeenCalledWith(
        "ANTHROPIC_API_KEY",
        "sk-ant-only",
        undefined,
      );
    });
    // No spec change → no settings write (and no double toast).
    expect(saveSettings).not.toHaveBeenCalled();
  });

  it("shows the 'already signed in' banner in the credentials section when authenticated", async () => {
    acpAuthStatusMock.mockReturnValue({
      status: "authenticated",
      isChecking: false,
      isSupported: true,
    });
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: [],
        },
      }),
    );

    renderAgentSettingsScreen();
    await screen.findByTestId("agent-settings-screen");

    expect(
      await screen.findByTestId("settings-acp-auth-detected"),
    ).toBeInTheDocument();
  });
});
