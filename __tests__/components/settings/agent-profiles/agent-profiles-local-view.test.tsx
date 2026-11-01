import { DEFAULT_ENABLED_SKILL_NAMES } from "@openhands/extensions/skills";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { useEffect } from "react";
import type { AgentSettingsSaveControl } from "#/routes/agent-settings";
import { AgentProfilesLocalView } from "#/components/features/settings/agent-profiles/agent-profiles-local-view";
import AgentProfilesSettingsRoute from "#/routes/agent-profiles-settings";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { ensureDefaultAgentProfileMigrated } from "#/hooks/use-migrate-default-agent-profile";

// The embedded Agent settings form is stubbed to emit a caller-provided
// control, so the tests exercise the view's mapping to AgentProfileSaveInput.
let emitControl: AgentSettingsSaveControl | null = null;
const ADD_PROFILE_LABEL = "add";
let editTarget = "my-agent";
const EDIT_PROFILE_LABEL = "edit";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

// The view imports the NAMED `AgentSettingsScreen` export (not the route's
// default, which React Router wraps and would strip the embedded props). Mock
// the named export to match; the factory is hoisted, so define the stub inline.
vi.mock("#/routes/agent-settings", () => {
  const MockAgentSettings = ({
    agentSettingsOverride,
    onSaveControlChange,
    onRouterEditingChange,
    isDefaultProfile,
    profileNameField,
    llmPicker,
  }: {
    agentSettingsOverride?: Record<string, unknown> | null;
    onSaveControlChange?: (c: AgentSettingsSaveControl) => void;
    onRouterEditingChange?: (editing: boolean) => void;
    isDefaultProfile?: boolean;
    profileNameField?: React.ReactNode;
    llmPicker?: React.ReactNode;
  }) => {
    useEffect(() => {
      if (emitControl) onSaveControlChange?.(emitControl);
    }, [onSaveControlChange]);
    // Surface the seed override so tests can assert what the embedded form
    // would open on.
    return (
      <div
        data-testid="mock-agent-settings"
        data-override={JSON.stringify(agentSettingsOverride)}
        data-default-profile={String(Boolean(isDefaultProfile))}
      >
        {profileNameField}
        {emitControl?.agentType === "acp" ? null : llmPicker}
        <button
          type="button"
          data-testid="test-router-edit"
          onClick={() => onRouterEditingChange?.(true)}
        />
        <button
          type="button"
          data-testid="test-router-done"
          onClick={() => onRouterEditingChange?.(false)}
        />
      </div>
    );
  };
  return {
    __esModule: true,
    AgentSettingsScreen: MockAgentSettings,
    default: MockAgentSettings,
  };
});

vi.mock(
  "#/components/features/settings/agent-profiles/agent-profiles-manager",
  () => ({
    AgentProfilesManager: ({
      onAddProfile,
      onEditProfile,
    }: {
      onAddProfile?: () => void;
      onEditProfile?: (profile: { name: string }) => void;
    }) => (
      <>
        <button
          type="button"
          data-testid="add-agent-profile"
          onClick={onAddProfile}
          aria-label={ADD_PROFILE_LABEL}
        />
        <button
          type="button"
          data-testid="edit-agent-profile"
          onClick={() => onEditProfile?.({ name: editTarget })}
          aria-label={EDIT_PROFILE_LABEL}
        />
      </>
    ),
  }),
);

const saveMutate = vi.fn().mockResolvedValue({ name: "x", message: "ok" });
vi.mock("#/hooks/mutation/use-save-agent-profile", () => ({
  useSaveAgentProfile: () => ({ mutateAsync: saveMutate }),
}));

const renameMutate = vi.fn().mockResolvedValue({ name: "x", message: "ok" });
vi.mock("#/hooks/mutation/use-rename-agent-profile", () => ({
  useRenameAgentProfile: () => ({ mutateAsync: renameMutate }),
}));

let agentProfilesData: {
  profiles: { name: string }[];
  active_agent_profile_id: string | null;
};
vi.mock("#/hooks/query/use-agent-profiles", () => ({
  useAgentProfiles: () => ({ data: agentProfilesData }),
}));

let llmProfilesData: {
  profiles: { name: string; model: string | null }[];
  active_profile: string | null;
};
vi.mock("#/hooks/query/use-llm-profiles", () => ({
  useLlmProfiles: () => ({ data: llmProfilesData }),
}));

let settingsData: {
  enabled_skills?: string[];
  disabled_skills?: string[];
  mcp_config?: Record<string, Record<string, unknown>>;
};
vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({ data: settingsData }),
}));

vi.mock("#/contexts/settings-section-header-context", () => ({
  useSettingsSectionHeader: () => ({ setHideSectionHeader: vi.fn() }),
}));

vi.mock("#/utils/custom-toast-handlers");

const profileSupportsSkillsMock = vi.hoisted(() => vi.fn(() => true));
vi.mock("#/api/agent-profiles-service/profile-field-support", () => ({
  agentProfileSupportsSkills: () => profileSupportsSkillsMock(),
}));

const applyToGlobalMock = vi.hoisted(() => vi.fn(async () => undefined));
vi.mock("#/hooks/mutation/sync-agent-profiles", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("#/hooks/mutation/sync-agent-profiles")
  >()),
  applyDefaultAgentProfileToGlobal: applyToGlobalMock,
}));

vi.mock("#/hooks/use-migrate-default-agent-profile", () => ({
  ensureDefaultAgentProfileMigrated: vi.fn(),
}));

const backendKind = vi.hoisted(() => ({ current: "local" }));
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "b1", kind: backendKind.current },
    orgId: null,
  }),
}));

function renderView() {
  return render(<AgentProfilesLocalView />, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={new QueryClient()}>
        {children}
      </QueryClientProvider>
    ),
  });
}

vi.mock("#/api/agent-profiles-service/agent-profiles-service.api", () => ({
  __esModule: true,
  default: { getProfile: vi.fn(), renameProfile: vi.fn() },
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME: "default",
}));

async function openCreateAndName(name: string) {
  const user = userEvent.setup();
  await user.click(screen.getByTestId("add-agent-profile"));
  await screen.findByTestId("mock-agent-settings");
  const input = screen.getByTestId("agent-profile-name-input");
  await user.clear(input);
  await user.type(input, name);
  return user;
}

describe("AgentProfilesLocalView save mapping", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(ensureDefaultAgentProfileMigrated).mockResolvedValue(true);
    emitControl = null;
    agentProfilesData = { profiles: [], active_agent_profile_id: null };
    llmProfilesData = {
      profiles: [{ name: "default", model: "gpt-5" }],
      active_profile: "default",
    };
    settingsData = {};
    editTarget = "my-agent";
    profileSupportsSkillsMock.mockReturnValue(true);
    backendKind.current = "local";
  });

  it("renders the Agents route as a standalone page without settings navigation", () => {
    render(<AgentProfilesSettingsRoute />, {
      wrapper: ({ children }) => (
        <QueryClientProvider client={new QueryClient()}>
          {children}
        </QueryClientProvider>
      ),
    });

    expect(screen.getAllByRole("main")).toHaveLength(1);
    expect(screen.getByTestId("add-agent-profile")).toBeInTheDocument();
    expect(screen.queryByTestId("settings-navbar-desktop")).toBeNull();
  });

  it("keeps the list focused on agent profiles rather than separate resource libraries", () => {
    renderView();

    expect(screen.getByTestId("add-agent-profile")).toBeInTheDocument();
    expect(screen.queryByTestId("agents-llm-library")).toBeNull();
    expect(screen.queryByTestId("agents-secrets-library")).toBeNull();
    expect(screen.queryByTestId("agents-skills-library")).toBeNull();
  });

  it("opens a new profile on the settings lists when there is no OpenHands default", async () => {
    settingsData = { enabled_skills: ["add-skill"], disabled_skills: ["mine"] };

    renderView();
    await openCreateAndName("my-agent");

    const override = JSON.parse(
      screen
        .getByTestId("mock-agent-settings")
        .getAttribute("data-override") as string,
    );
    expect(override).toMatchObject({
      agent_kind: "openhands",
      public_skills: ["add-skill"],
      disabled_skills: ["mine"],
    });
  });

  it("prevents agent save until the inline router editor finishes", async () => {
    emitControl = {
      agentType: "openhands",
      isValid: true,
      isDirty: true,
      buildAgentProfileFields: () => ({
        agent_kind: "openhands",
        mcp_server_refs: null,
      }),
      credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
    };
    renderView();
    const user = await openCreateAndName("router-agent");
    const save = screen.getByTestId("save-agent-profile-btn");
    expect(save).toBeEnabled();

    await user.click(screen.getByTestId("test-router-edit"));
    expect(save).toBeDisabled();
    await user.click(screen.getByTestId("test-router-done"));
    expect(save).toBeEnabled();
  });

  it("opens a new profile without a skill list on a cloud backend", async () => {
    profileSupportsSkillsMock.mockReturnValue(false);
    settingsData = { enabled_skills: ["add-skill"], disabled_skills: ["mine"] };

    renderView();
    await openCreateAndName("my-agent");

    expect(
      JSON.parse(
        screen
          .getByTestId("mock-agent-settings")
          .getAttribute("data-override") as string,
      ),
    ).toEqual({ agent_kind: "openhands" });
  });

  it("starts a new profile from default, skills included, with every server and secret", async () => {
    settingsData = { enabled_skills: ["add-skill"], disabled_skills: [] };
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      name: "default",
      profile: {
        id: "p-default",
        name: "default",
        revision: 4,
        agent_kind: "openhands",
        llm_profile_ref: "other-llm",
        tools: [{ name: "terminal", params: {} }],
        tool_concurrency_limit: 3,
        system_message_suffix: "Be terse.",
        condenser: { kind: "NoOpCondenserSettings" },
        verification: { critic_enabled: true },
        enable_classify_and_switch_llm_tool: true,
        meta_profile_ref: "fast-router",
        mcp_server_refs: ["github"],
        secret_refs: ["GITHUB_TOKEN"],
        public_skills: ["only-in-default"],
        disabled_skills: ["off-in-default"],
      },
    } as never);
    emitControl = {
      agentType: "openhands",
      isValid: true,
      isDirty: true,
      buildAgentProfileFields: () => ({
        agent_kind: "openhands",
        mcp_server_refs: null,
        public_skills: ["only-in-default"],
        disabled_skills: ["off-in-default"],
      }),
      credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
    };

    renderView();
    const user = await openCreateAndName("my-agent");

    expect(AgentProfilesService.getProfile).toHaveBeenCalledWith("default");
    const override = JSON.parse(
      screen
        .getByTestId("mock-agent-settings")
        .getAttribute("data-override") as string,
    );
    expect(override).toMatchObject({
      agent_kind: "openhands",
      tools: [{ name: "terminal", params: {} }],
      tool_concurrency_limit: 3,
      system_message_suffix: "Be terse.",
      condenser: { kind: "NoOpCondenserSettings" },
      verification: { critic_enabled: true },
      enable_classify_and_switch_llm_tool: true,
      meta_profile_ref: "fast-router",
      mcp_server_refs: null,
      secret_refs: null,
      public_skills: ["only-in-default"],
      disabled_skills: ["off-in-default"],
    });

    await user.click(screen.getByTestId("save-agent-profile-btn"));
    await waitFor(() => expect(saveMutate).toHaveBeenCalledTimes(1));
    const { profile } = saveMutate.mock.calls[0][0];
    expect(profile).toEqual({
      agent_kind: "openhands",
      llm_profile_ref: "default",
      tools: [{ name: "terminal", params: {} }],
      tool_concurrency_limit: 3,
      system_message_suffix: "Be terse.",
      condenser: { kind: "NoOpCondenserSettings" },
      verification: { critic_enabled: true },
      enable_classify_and_switch_llm_tool: true,
      meta_profile_ref: "fast-router",
      mcp_server_refs: null,
      public_skills: ["only-in-default"],
      disabled_skills: ["off-in-default"],
    });
  });

  it("starts a new profile without default's settings when default is an ACP agent", async () => {
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      name: "default",
      profile: {
        id: "p-default",
        name: "default",
        revision: 1,
        agent_kind: "acp",
        acp_server: "claude-code",
      },
    } as never);

    renderView();
    await openCreateAndName("my-agent");

    expect(
      JSON.parse(
        screen
          .getByTestId("mock-agent-settings")
          .getAttribute("data-override") as string,
      ),
    ).toEqual({
      agent_kind: "openhands",
      public_skills: [...DEFAULT_ENABLED_SKILL_NAMES],
      disabled_skills: [],
    });
  });

  it("waits for typing before calling a new profile's empty name invalid", async () => {
    renderView();
    const user = await openCreateAndName("x");
    const input = screen.getByTestId("agent-profile-name-input");
    await user.clear(input);

    expect(input).toHaveAttribute("aria-invalid", "true");
  });

  it("starts a new profile without flagging its empty name", async () => {
    renderView();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("add-agent-profile"));
    await screen.findByTestId("mock-agent-settings");

    expect(screen.getByTestId("agent-profile-name-input")).toHaveAttribute(
      "aria-invalid",
      "false",
    );
    expect(screen.getByTestId("save-agent-profile-btn")).toBeDisabled();
  });

  it("describes the add editor with agent copy, not the LLM-profile hint", async () => {
    renderView();
    await openCreateAndName("my-agent");

    expect(
      screen.getByTestId("agent-profile-editor-description"),
    ).toHaveTextContent("SETTINGS$AGENT_PROFILE_SAVE_HINT");
  });

  it("saves an OpenHands profile with the selected llm_profile_ref", async () => {
    emitControl = {
      agentType: "openhands",
      isValid: true,
      isDirty: true,
      buildAgentProfileFields: () => ({
        agent_kind: "openhands",
        mcp_server_refs: null,
      }),
      credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
    };

    renderView();
    const user = await openCreateAndName("my-oh");
    await user.click(screen.getByTestId("save-agent-profile-btn"));

    await waitFor(() => expect(saveMutate).toHaveBeenCalledTimes(1));
    expect(saveMutate).toHaveBeenCalledWith({
      name: "my-oh",
      profile: {
        agent_kind: "openhands",
        mcp_server_refs: null,
        llm_profile_ref: "default",
        public_skills: [...DEFAULT_ENABLED_SKILL_NAMES],
      },
    });
  });

  it("saves an ACP profile without an llm_profile_ref", async () => {
    emitControl = {
      agentType: "acp",
      isValid: true,
      isDirty: true,
      buildAgentProfileFields: () => ({
        agent_kind: "acp",
        mcp_server_refs: null,
        acp_server: "claude-code",
        acp_model: "claude-opus-4-8",
        acp_command: null,
        acp_args: null,
      }),
      credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
    };

    renderView();
    const user = await openCreateAndName("my-claude");
    await user.click(screen.getByTestId("save-agent-profile-btn"));

    await waitFor(() => expect(saveMutate).toHaveBeenCalledTimes(1));
    expect(saveMutate).toHaveBeenCalledWith({
      name: "my-claude",
      profile: {
        agent_kind: "acp",
        mcp_server_refs: null,
        acp_server: "claude-code",
        acp_model: "claude-opus-4-8",
        acp_command: null,
        acp_args: null,
      },
    });
  });

  it("edit-save round-trips stored fields the editor doesn't model", async () => {
    // The seeded `default` profile carries fields the minimal editor never
    // shows; the save is a whole-profile overwrite, so they must ride the
    // payload untouched (with server-managed identity stripped).
    const storedProfile = {
      schema_version: 1,
      id: "p-1",
      name: "my-agent",
      revision: 3,
      agent_kind: "openhands",
      llm_profile_ref: "default",
      agent: "CodeActAgent",
      system_message_suffix: "Be terse.",
      condenser: { kind: "NoOpCondenserSettings" },
      verification: { critic_enabled: true },
      tools: [{ name: "terminal", params: {} }],
      tool_concurrency_limit: 4,
      mcp_server_refs: ["github"],
      disabled_skills: ["deploy-checklist"],
      enable_classify_and_switch_llm_tool: true,
      meta_profile_ref: "fast-router",
    };
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      name: "my-agent",
      profile: storedProfile,
    } as never);
    emitControl = {
      agentType: "openhands",
      isValid: true,
      isDirty: true,
      // The editor models `mcp_server_refs`, so it re-emits what it was seeded
      // with; `disabled_skills` it does not, and survives via the merge.
      buildAgentProfileFields: () => ({
        agent_kind: "openhands",
        mcp_server_refs: ["github"],
        tools: [{ name: "grep", params: {} }],
      }),
      credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
    };

    renderView();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("edit-agent-profile"));
    await screen.findByTestId("mock-agent-settings");

    // Asserted before the save: a successful save returns to the list and
    // unmounts the form.
    const seededOverride = JSON.parse(
      screen
        .getByTestId("mock-agent-settings")
        .getAttribute("data-override") as string,
    );
    expect(seededOverride).toMatchObject({
      agent_kind: "openhands",
      tools: [{ name: "terminal", params: {} }],
      tool_concurrency_limit: 4,
      // Without this the picker opens on "all servers" and the save widens the
      // profile's scope back to every configured server.
      mcp_server_refs: ["github"],
      secret_refs: null,
      // The editor's Condenser and Verification sections open on these.
      condenser: { kind: "NoOpCondenserSettings" },
      verification: { critic_enabled: true },
      // The editor's Model Router section opens on these.
      enable_classify_and_switch_llm_tool: true,
      meta_profile_ref: "fast-router",
    });

    await user.click(screen.getByTestId("save-agent-profile-btn"));

    await waitFor(() => expect(saveMutate).toHaveBeenCalledTimes(1));
    // Profiles are secret-free now (no embedded skills), so exposeSecrets is a
    // server-side no-op; canvas still passes "encrypted" for signature parity.
    expect(AgentProfilesService.getProfile).toHaveBeenCalledWith(
      "my-agent",
      "encrypted",
    );
    const { profile } = saveMutate.mock.calls[0][0];
    expect(profile).toMatchObject({
      agent_kind: "openhands",
      tools: [{ name: "grep", params: {} }],
      llm_profile_ref: "default",
      system_message_suffix: "Be terse.",
      condenser: { kind: "NoOpCondenserSettings" },
      verification: { critic_enabled: true },
      tool_concurrency_limit: 4,
      mcp_server_refs: ["github"],
      disabled_skills: ["deploy-checklist"],
      enable_classify_and_switch_llm_tool: true,
      meta_profile_ref: "fast-router",
    });
    expect(profile).not.toHaveProperty("id");
    expect(profile).not.toHaveProperty("name");
    expect(profile).not.toHaveProperty("revision");
    expect(applyToGlobalMock).not.toHaveBeenCalled();
  });

  describe("the default profile", () => {
    const OPTIONAL = "add-javadoc";
    const storedDefault = {
      id: "p-default",
      name: "default",
      revision: 2,
      agent_kind: "openhands",
      llm_profile_ref: "stale",
      mcp_server_refs: ["github"],
      secret_refs: ["GITHUB_TOKEN"],
      public_skills: ["github"],
      disabled_skills: [],
    };

    beforeEach(() => {
      editTarget = "default";
      llmProfilesData = {
        profiles: [
          { name: "stale", model: "gpt-4" },
          { name: "fast", model: "gpt-5" },
        ],
        active_profile: "fast",
      };
      settingsData = {
        enabled_skills: [OPTIONAL],
        disabled_skills: ["mine"],
        mcp_config: {
          github: { transport: "http", url: "https://github.example/mcp" },
          slack: { transport: "stdio", command: "npx", enabled: false },
        },
      };
      vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
        name: "default",
        profile: storedDefault,
      } as never);
      emitControl = {
        agentType: "openhands",
        isValid: true,
        isDirty: true,
        buildAgentProfileFields: () => ({
          agent_kind: "openhands",
          mcp_server_refs: ["slack"],
          secret_refs: null,
          public_skills: [OPTIONAL],
          disabled_skills: [],
        }),
        credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
      };
    });

    it("opens on the MCP servers switched on, every secret, and its own skills", async () => {
      renderView();
      const user = userEvent.setup();
      await user.click(screen.getByTestId("edit-agent-profile"));
      const form = await screen.findByTestId("mock-agent-settings");

      expect(form).toHaveAttribute("data-default-profile", "true");
      expect(
        JSON.parse(form.getAttribute("data-override") as string),
      ).toMatchObject({
        mcp_server_refs: ["github"],
        secret_refs: null,
        public_skills: ["github"],
        disabled_skills: [],
      });
      expect(
        screen.getByTestId("agent-profile-llm-default-note"),
      ).toBeInTheDocument();
    });

    it("writes its edits to the global pages and stores every server and secret", async () => {
      renderView();
      const user = userEvent.setup();
      await user.click(screen.getByTestId("edit-agent-profile"));
      await screen.findByTestId("mock-agent-settings");
      await user.click(screen.getByTestId("save-agent-profile-btn"));

      await waitFor(() => expect(saveMutate).toHaveBeenCalledTimes(1));
      expect(applyToGlobalMock).toHaveBeenCalledWith(
        expect.anything(),
        expect.objectContaining({
          llm_profile_ref: "fast",
          mcp_server_refs: ["slack"],
          public_skills: [OPTIONAL],
        }),
      );
      const { name, profile } = saveMutate.mock.calls[0][0];
      expect(name).toBe("default");
      expect(profile).toMatchObject({
        llm_profile_ref: "fast",
        mcp_server_refs: null,
        secret_refs: null,
      });
      // The profile saves first, so a rejected save changes no global page.
      expect(saveMutate.mock.invocationCallOrder[0]).toBeLessThan(
        applyToGlobalMock.mock.invocationCallOrder[0],
      );
    });

    it("changes no global page when the profile save fails", async () => {
      saveMutate.mockRejectedValueOnce(new Error("conflict"));
      renderView();
      const user = userEvent.setup();
      await user.click(screen.getByTestId("edit-agent-profile"));
      await screen.findByTestId("mock-agent-settings");
      await user.click(screen.getByTestId("save-agent-profile-btn"));

      await waitFor(() => expect(displayErrorToast).toHaveBeenCalled());
      expect(applyToGlobalMock).not.toHaveBeenCalled();
    });

    it("treats an ACP default switched to OpenHands as the shared default", async () => {
      vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
        name: "default",
        profile: {
          ...storedDefault,
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: "claude",
        },
      } as never);
      emitControl = {
        agentType: "openhands",
        isValid: true,
        isDirty: true,
        buildAgentProfileFields: () => ({
          agent_kind: "openhands",
          mcp_server_refs: ["github"],
          secret_refs: ["GITHUB_TOKEN"],
        }),
        credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
      };

      renderView();
      const user = userEvent.setup();
      await user.click(screen.getByTestId("edit-agent-profile"));
      const form = await screen.findByTestId("mock-agent-settings");

      expect(form).toHaveAttribute("data-default-profile", "true");
      expect(
        screen.getByTestId("agent-profile-llm-default-note"),
      ).toBeInTheDocument();
      await user.click(screen.getByTestId("save-agent-profile-btn"));
      await waitFor(() => expect(applyToGlobalMock).toHaveBeenCalled());
      expect(saveMutate.mock.calls[0][0].profile).toMatchObject({
        agent_kind: "openhands",
        llm_profile_ref: "fast",
        mcp_server_refs: null,
        secret_refs: null,
      });
    });

    it("leaves an ACP default's own servers and secrets alone", async () => {
      vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
        name: "default",
        profile: {
          ...storedDefault,
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: "claude",
        },
      } as never);
      emitControl = {
        agentType: "acp",
        isValid: true,
        isDirty: true,
        buildAgentProfileFields: () => ({
          agent_kind: "acp",
          mcp_server_refs: ["github"],
          secret_refs: ["GITHUB_TOKEN"],
          acp_server: "claude-code",
          acp_model: null,
          acp_command: "claude",
          acp_args: null,
        }),
        credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
      };

      renderView();
      const user = userEvent.setup();
      await user.click(screen.getByTestId("edit-agent-profile"));
      const form = await screen.findByTestId("mock-agent-settings");

      // The editor hides the shared controls only for an OpenHands agent.
      expect(form).toHaveAttribute("data-default-profile", "true");
      expect(
        JSON.parse(form.getAttribute("data-override") as string),
      ).toMatchObject({
        mcp_server_refs: ["github"],
        secret_refs: ["GITHUB_TOKEN"],
      });
      expect(
        screen.queryByTestId("agent-profile-llm-default-note"),
      ).not.toBeInTheDocument();
      await user.click(screen.getByTestId("save-agent-profile-btn"));
      await waitFor(() => expect(saveMutate).toHaveBeenCalledTimes(1));
      expect(applyToGlobalMock).not.toHaveBeenCalled();
      expect(saveMutate.mock.calls[0][0].profile).toMatchObject({
        secret_refs: ["GITHUB_TOKEN"],
      });
    });

    it("edits a cloud profile named default like any other profile", async () => {
      backendKind.current = "cloud";
      renderView();
      const user = userEvent.setup();
      await user.click(screen.getByTestId("edit-agent-profile"));
      const form = await screen.findByTestId("mock-agent-settings");

      expect(form).toHaveAttribute("data-default-profile", "false");
      expect(screen.getByTestId("agent-profile-name-input")).toBeEnabled();
      await user.click(screen.getByTestId("save-agent-profile-btn"));
      await waitFor(() => expect(saveMutate).toHaveBeenCalledTimes(1));
      expect(applyToGlobalMock).not.toHaveBeenCalled();
      expect(saveMutate.mock.calls[0][0].profile).toMatchObject({
        llm_profile_ref: "stale",
        mcp_server_refs: ["slack"],
      });
    });

    it("locks default's name, which the server would otherwise reseed", async () => {
      renderView();
      const user = userEvent.setup();
      await user.click(screen.getByTestId("edit-agent-profile"));
      await screen.findByTestId("mock-agent-settings");

      expect(screen.getByTestId("agent-profile-name-input")).toBeDisabled();
      expect(
        screen.getByTestId("agent-profile-default-locked-note"),
      ).toHaveTextContent("SETTINGS$AGENT_PROFILE_DEFAULT_LOCKED");
    });
  });

  it("seeds tools from a stored profile", async () => {
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      name: "my-agent",
      profile: {
        schema_version: 3,
        id: "p-1",
        name: "my-agent",
        revision: 3,
        agent_kind: "openhands",
        llm_profile_ref: "default",
        tools: [{ name: "terminal" }, { name: "task_tool_set" }],
      },
    } as never);
    emitControl = {
      agentType: "openhands",
      isValid: true,
      isDirty: false,
      buildAgentProfileFields: () => ({
        agent_kind: "openhands",
        mcp_server_refs: null,
      }),
      credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
    };

    renderView();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("edit-agent-profile"));
    await screen.findByTestId("mock-agent-settings");

    const seededOverride = JSON.parse(
      screen
        .getByTestId("mock-agent-settings")
        .getAttribute("data-override") as string,
    );
    expect(seededOverride).toMatchObject({
      tools: [{ name: "terminal" }, { name: "task_tool_set" }],
    });
  });

  it("seeds the editor with a stored secret scope", async () => {
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      name: "my-agent",
      profile: {
        schema_version: 2,
        id: "p-1",
        name: "my-agent",
        revision: 3,
        agent_kind: "openhands",
        llm_profile_ref: "default",
        secret_refs: ["DATADOG_API_KEY"],
      },
    } as never);
    emitControl = {
      agentType: "openhands",
      isValid: true,
      isDirty: true,
      buildAgentProfileFields: () => ({
        agent_kind: "openhands",
        mcp_server_refs: null,
      }),
      credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
    };

    renderView();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("edit-agent-profile"));
    await screen.findByTestId("mock-agent-settings");

    const seededOverride = JSON.parse(
      screen
        .getByTestId("mock-agent-settings")
        .getAttribute("data-override") as string,
    );
    expect(seededOverride.secret_refs).toEqual(["DATADOG_API_KEY"]);
  });

  it("seeds the editor with a stored persona and instructions", async () => {
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      name: "my-agent",
      profile: {
        schema_version: 2,
        id: "p-1",
        name: "my-agent",
        revision: 3,
        agent_kind: "openhands",
        llm_profile_ref: "default",
        persona: "You triage issues.",
        system_message_suffix: "Be terse.",
      },
    } as never);
    emitControl = {
      agentType: "openhands",
      isValid: true,
      isDirty: false,
      buildAgentProfileFields: () => ({
        agent_kind: "openhands",
        mcp_server_refs: null,
      }),
      credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
    };

    renderView();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("edit-agent-profile"));
    await screen.findByTestId("mock-agent-settings");

    const seededOverride = JSON.parse(
      screen
        .getByTestId("mock-agent-settings")
        .getAttribute("data-override") as string,
    );
    expect(seededOverride.persona).toBe("You triage issues.");
    expect(seededOverride.system_message_suffix).toBe("Be terse.");
  });

  it("edit-save persists an edited secret scope over the stored value", async () => {
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      name: "my-agent",
      profile: {
        schema_version: 2,
        id: "p-1",
        name: "my-agent",
        revision: 3,
        agent_kind: "openhands",
        llm_profile_ref: "default",
        secret_refs: null,
      },
    } as never);
    emitControl = {
      agentType: "openhands",
      isValid: true,
      isDirty: true,
      buildAgentProfileFields: () => ({
        agent_kind: "openhands",
        mcp_server_refs: null,
        secret_refs: ["DATADOG_API_KEY"],
      }),
      credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
    };

    renderView();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("edit-agent-profile"));
    await screen.findByTestId("mock-agent-settings");
    await user.click(screen.getByTestId("save-agent-profile-btn"));

    await waitFor(() => expect(saveMutate).toHaveBeenCalledTimes(1));
    const { profile } = saveMutate.mock.calls[0][0];
    expect(profile.secret_refs).toEqual(["DATADOG_API_KEY"]);
  });

  it("kind-switch edit-save sends a clean variant payload", async () => {
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      name: "my-agent",
      profile: {
        schema_version: 1,
        id: "p-1",
        name: "my-agent",
        revision: 3,
        agent_kind: "openhands",
        llm_profile_ref: "default",
        condenser: { kind: "NoOpCondenserSettings" },
        mcp_server_refs: ["github"],
      },
    } as never);
    emitControl = {
      agentType: "acp",
      isValid: true,
      isDirty: true,
      buildAgentProfileFields: () => ({
        agent_kind: "acp",
        mcp_server_refs: null,
        acp_server: "claude-code",
        acp_model: "claude-opus-4-8",
        acp_command: null,
        acp_args: null,
      }),
      credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
    };

    renderView();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("edit-agent-profile"));
    await screen.findByTestId("mock-agent-settings");
    await user.click(screen.getByTestId("save-agent-profile-btn"));

    await waitFor(() => expect(saveMutate).toHaveBeenCalledTimes(1));
    // No stored openhands fields may leak into the acp payload — the server's
    // extra="forbid" union would 422 on a mongrel profile.
    expect(saveMutate).toHaveBeenCalledWith({
      name: "my-agent",
      profile: {
        agent_kind: "acp",
        mcp_server_refs: null,
        acp_server: "claude-code",
        acp_model: "claude-opus-4-8",
        acp_command: null,
        acp_args: null,
      },
    });
  });

  it("falls back to the default LLM profile when the stored llm_profile_ref is dangling (#1571 review)", async () => {
    // The stored profile references an LLM profile that's since been deleted;
    // the editor must validate against the live list and self-heal to the
    // active default rather than saving the stale ref straight back.
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      name: "my-agent",
      profile: {
        schema_version: 1,
        id: "p-1",
        name: "my-agent",
        revision: 3,
        agent_kind: "openhands",
        llm_profile_ref: "deleted-profile",
      },
    } as never);
    emitControl = {
      agentType: "openhands",
      isValid: true,
      isDirty: false,
      buildAgentProfileFields: () => ({
        agent_kind: "openhands",
        mcp_server_refs: null,
      }),
      credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
    };

    renderView();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("edit-agent-profile"));
    await screen.findByTestId("mock-agent-settings");
    await waitFor(() => {
      expect(screen.getByTestId("save-agent-profile-btn")).not.toBeDisabled();
    });
    await user.click(screen.getByTestId("save-agent-profile-btn"));

    await waitFor(() => expect(saveMutate).toHaveBeenCalledTimes(1));
    const { profile } = saveMutate.mock.calls[0][0];
    expect(profile.llm_profile_ref).toBe("default");
  });

  it("keeps a live llm_profile_ref untouched on load", async () => {
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      name: "custom",
      profile: {
        schema_version: 1,
        id: "p-2",
        name: "custom",
        revision: 1,
        agent_kind: "openhands",
        llm_profile_ref: "default",
      },
    } as never);
    emitControl = {
      agentType: "openhands",
      isValid: true,
      isDirty: false,
      buildAgentProfileFields: () => ({
        agent_kind: "openhands",
        mcp_server_refs: null,
      }),
      credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
    };

    renderView();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("edit-agent-profile"));
    await screen.findByTestId("mock-agent-settings");
    expect(screen.getByTestId("save-agent-profile-btn")).toBeDisabled();
  });

  it("keeps the agent draft while creating a reusable LLM profile inline", async () => {
    llmProfilesData = { profiles: [], active_profile: null };
    renderView();
    const user = await openCreateAndName("my-agent");

    await user.click(
      screen.getByTestId("agent-profile-add-llm").querySelector("summary")!,
    );

    expect(await screen.findByTestId("agent-inline-llm-editor")).toBeVisible();
    expect(screen.queryByTestId("back-to-profiles")).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("profile-editor-title"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("agent-profile-name-input")).toHaveValue(
      "my-agent",
    );
    expect(screen.getByTestId("save-agent-profile-btn")).toBeDisabled();
  });

  it("blocks an OpenHands save when no LLM profile is available", async () => {
    llmProfilesData = { profiles: [], active_profile: null };
    emitControl = {
      agentType: "openhands",
      isValid: true,
      isDirty: true,
      buildAgentProfileFields: () => ({
        agent_kind: "openhands",
        mcp_server_refs: null,
      }),
      credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
    };

    renderView();
    const user = await openCreateAndName("my-oh");
    await user.click(screen.getByTestId("save-agent-profile-btn"));

    await waitFor(() =>
      expect(displayErrorToast).toHaveBeenCalledWith(
        "SETTINGS$AGENT_PROFILE_LLM_REQUIRED",
      ),
    );
    expect(saveMutate).not.toHaveBeenCalled();
  });

  it("reads a profile to edit only after the default migration", async () => {
    let finishMigration: (value: true) => void = () => {};
    vi.mocked(ensureDefaultAgentProfileMigrated).mockReturnValue(
      new Promise((resolve) => {
        finishMigration = resolve;
      }),
    );
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      name: "my-agent",
      profile: {
        id: "p-1",
        name: "my-agent",
        revision: 1,
        agent_kind: "openhands",
        llm_profile_ref: "default",
      },
    } as never);

    renderView();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("edit-agent-profile"));

    expect(ensureDefaultAgentProfileMigrated).toHaveBeenCalled();
    expect(AgentProfilesService.getProfile).not.toHaveBeenCalled();
    finishMigration(true);
    await screen.findByTestId("mock-agent-settings");
    expect(AgentProfilesService.getProfile).toHaveBeenCalledWith(
      "my-agent",
      "encrypted",
    );
  });

  describe("duplicate profile name", () => {
    beforeEach(() => {
      agentProfilesData = {
        profiles: [{ name: "my-agent" }, { name: "triage" }],
        active_agent_profile_id: null,
      };
      emitControl = {
        agentType: "openhands",
        isValid: true,
        isDirty: false,
        buildAgentProfileFields: () => ({
          agent_kind: "openhands",
          mcp_server_refs: null,
        }),
        credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
      };
    });

    it("tells the user a new profile's name is taken until it is unique", async () => {
      renderView();
      const user = await openCreateAndName("my-agent");
      const input = screen.getByTestId("agent-profile-name-input");

      expect(input).toHaveAttribute("aria-invalid", "true");
      expect(input).toHaveAccessibleDescription(
        "SETTINGS$AGENT_PROFILE_NAME_TAKEN",
      );
      expect(screen.getByRole("alert")).toHaveTextContent(
        "SETTINGS$AGENT_PROFILE_NAME_TAKEN",
      );
      expect(screen.getByTestId("save-agent-profile-btn")).toBeDisabled();

      await user.type(input, "-2");

      expect(input).toHaveAttribute("aria-invalid", "false");
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expect(screen.getByTestId("save-agent-profile-btn")).toBeEnabled();
    });

    it("flags a rename to another profile's name but not the profile's own name", async () => {
      vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
        name: "my-agent",
        profile: {
          schema_version: 3,
          id: "p-1",
          name: "my-agent",
          revision: 1,
          agent_kind: "openhands",
          llm_profile_ref: "default",
        },
      } as never);

      renderView();
      const user = userEvent.setup();
      await user.click(screen.getByTestId("edit-agent-profile"));
      await screen.findByTestId("mock-agent-settings");
      const input = screen.getByTestId("agent-profile-name-input");

      expect(input).toHaveValue("my-agent");
      expect(input).toHaveAttribute("aria-invalid", "false");
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();

      await user.clear(input);
      await user.type(input, "triage");

      expect(input).toHaveAttribute("aria-invalid", "true");
      expect(input).toHaveAccessibleDescription(
        "SETTINGS$AGENT_PROFILE_NAME_TAKEN",
      );
      expect(screen.getByTestId("save-agent-profile-btn")).toBeDisabled();
    });
  });
});
