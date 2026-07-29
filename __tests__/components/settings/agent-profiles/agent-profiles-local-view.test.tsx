import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { useEffect } from "react";
import type { AgentSettingsSaveControl } from "#/routes/agent-settings";
import { AgentProfilesLocalView } from "#/components/features/settings/agent-profiles/agent-profiles-local-view";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

// The embedded Agent settings form is stubbed to emit a caller-provided
// control, so the tests exercise the view's mapping to AgentProfileSaveInput.
let emitControl: AgentSettingsSaveControl | null = null;

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

// The view imports the NAMED `AgentSettingsScreen` export (not the route's
// default, which React Router wraps and would strip the embedded props). Mock
// the named export to match; the factory is hoisted, so define the stub inline.
vi.mock("#/routes/agent-settings", () => {
  const MockAgentSettings = ({
    onSaveControlChange,
  }: {
    onSaveControlChange?: (c: AgentSettingsSaveControl) => void;
  }) => {
    useEffect(() => {
      if (emitControl) onSaveControlChange?.(emitControl);
    }, [onSaveControlChange]);
    return <div data-testid="mock-agent-settings" />;
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
          aria-label="add"
        />
        <button
          type="button"
          data-testid="edit-agent-profile"
          onClick={() => onEditProfile?.({ name: "default" })}
          aria-label="edit"
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

const agentProfilesData = { profiles: [], active_agent_profile_id: null };
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

vi.mock("#/contexts/settings-section-header-context", () => ({
  useSettingsSectionHeader: () => ({ setHideSectionHeader: vi.fn() }),
}));

vi.mock("#/utils/custom-toast-handlers");

vi.mock("#/api/agent-profiles-service/agent-profiles-service.api", () => ({
  __esModule: true,
  default: { getProfile: vi.fn(), renameProfile: vi.fn() },
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
    emitControl = null;
    llmProfilesData = {
      profiles: [{ name: "default", model: "gpt-5" }],
      active_profile: "default",
    };
  });

  it("saves an OpenHands profile with the selected llm_profile_ref", async () => {
    emitControl = {
      agentType: "openhands",
      isValid: true,
      buildAgentProfileFields: () => ({
        agent_kind: "openhands",
        enable_sub_agents: true,
      }),
      credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
    };

    render(<AgentProfilesLocalView />);
    const user = await openCreateAndName("my-oh");
    await user.click(screen.getByTestId("save-agent-profile-btn"));

    await waitFor(() => expect(saveMutate).toHaveBeenCalledTimes(1));
    expect(saveMutate).toHaveBeenCalledWith({
      name: "my-oh",
      profile: {
        agent_kind: "openhands",
        enable_sub_agents: true,
        llm_profile_ref: "default",
      },
    });
  });

  it("saves an ACP profile without an llm_profile_ref", async () => {
    emitControl = {
      agentType: "acp",
      isValid: true,
      buildAgentProfileFields: () => ({
        agent_kind: "acp",
        acp_server: "claude-code",
        acp_model: "claude-opus-4-8",
        acp_command: null,
        acp_args: null,
      }),
      credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
    };

    render(<AgentProfilesLocalView />);
    const user = await openCreateAndName("my-claude");
    await user.click(screen.getByTestId("save-agent-profile-btn"));

    await waitFor(() => expect(saveMutate).toHaveBeenCalledTimes(1));
    expect(saveMutate).toHaveBeenCalledWith({
      name: "my-claude",
      profile: {
        agent_kind: "acp",
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
      name: "default",
      revision: 3,
      agent_kind: "openhands",
      llm_profile_ref: "default",
      agent: "CodeActAgent",
      system_message_suffix: "Be terse.",
      condenser: { kind: "NoOpCondenserSettings" },
      verification: { critic_enabled: true },
      enable_sub_agents: false,
      enable_switch_llm_tool: false,
      tool_concurrency_limit: 4,
      mcp_server_refs: ["github"],
      disabled_skills: ["deploy-checklist"],
    };
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      name: "default",
      profile: storedProfile,
    } as never);
    emitControl = {
      agentType: "openhands",
      isValid: true,
      buildAgentProfileFields: () => ({
        agent_kind: "openhands",
        enable_sub_agents: true,
      }),
      credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
    };

    render(<AgentProfilesLocalView />);
    const user = userEvent.setup();
    await user.click(screen.getByTestId("edit-agent-profile"));
    await screen.findByTestId("mock-agent-settings");
    await user.click(screen.getByTestId("save-agent-profile-btn"));

    await waitFor(() => expect(saveMutate).toHaveBeenCalledTimes(1));
    // Profiles are secret-free now (no embedded skills), so exposeSecrets is a
    // server-side no-op; canvas still passes "encrypted" for signature parity.
    expect(AgentProfilesService.getProfile).toHaveBeenCalledWith(
      "default",
      "encrypted",
    );
    const { profile } = saveMutate.mock.calls[0][0];
    expect(profile).toMatchObject({
      agent_kind: "openhands",
      enable_sub_agents: true,
      llm_profile_ref: "default",
      system_message_suffix: "Be terse.",
      condenser: { kind: "NoOpCondenserSettings" },
      verification: { critic_enabled: true },
      enable_switch_llm_tool: false,
      tool_concurrency_limit: 4,
      mcp_server_refs: ["github"],
      disabled_skills: ["deploy-checklist"],
    });
    expect(profile).not.toHaveProperty("id");
    expect(profile).not.toHaveProperty("name");
    expect(profile).not.toHaveProperty("revision");
  });

  it("kind-switch edit-save sends a clean variant payload", async () => {
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      name: "default",
      profile: {
        schema_version: 1,
        id: "p-1",
        name: "default",
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
      buildAgentProfileFields: () => ({
        agent_kind: "acp",
        acp_server: "claude-code",
        acp_model: "claude-opus-4-8",
        acp_command: null,
        acp_args: null,
      }),
      credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
    };

    render(<AgentProfilesLocalView />);
    const user = userEvent.setup();
    await user.click(screen.getByTestId("edit-agent-profile"));
    await screen.findByTestId("mock-agent-settings");
    await user.click(screen.getByTestId("save-agent-profile-btn"));

    await waitFor(() => expect(saveMutate).toHaveBeenCalledTimes(1));
    // No stored openhands fields may leak into the acp payload — the server's
    // extra="forbid" union would 422 on a mongrel profile.
    expect(saveMutate).toHaveBeenCalledWith({
      name: "default",
      profile: {
        agent_kind: "acp",
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
      name: "default",
      profile: {
        schema_version: 1,
        id: "p-1",
        name: "default",
        revision: 3,
        agent_kind: "openhands",
        llm_profile_ref: "deleted-profile",
        enable_sub_agents: false,
      },
    } as never);
    emitControl = {
      agentType: "openhands",
      isValid: true,
      buildAgentProfileFields: () => ({
        agent_kind: "openhands",
        enable_sub_agents: false,
      }),
      credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
    };

    render(<AgentProfilesLocalView />);
    const user = userEvent.setup();
    await user.click(screen.getByTestId("edit-agent-profile"));
    await screen.findByTestId("mock-agent-settings");
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
        enable_sub_agents: false,
      },
    } as never);
    emitControl = {
      agentType: "openhands",
      isValid: true,
      buildAgentProfileFields: () => ({
        agent_kind: "openhands",
        enable_sub_agents: false,
      }),
      credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
    };

    render(<AgentProfilesLocalView />);
    const user = userEvent.setup();
    await user.click(screen.getByTestId("edit-agent-profile"));
    await screen.findByTestId("mock-agent-settings");
    await user.click(screen.getByTestId("save-agent-profile-btn"));

    await waitFor(() => expect(saveMutate).toHaveBeenCalledTimes(1));
    const { profile } = saveMutate.mock.calls[0][0];
    expect(profile.llm_profile_ref).toBe("default");
  });

  it("blocks an OpenHands save when no LLM profile is available", async () => {
    llmProfilesData = { profiles: [], active_profile: null };
    emitControl = {
      agentType: "openhands",
      isValid: true,
      buildAgentProfileFields: () => ({
        agent_kind: "openhands",
        enable_sub_agents: false,
      }),
      credentials: { isDirty: false, save: vi.fn(), reset: vi.fn() },
    };

    render(<AgentProfilesLocalView />);
    const user = await openCreateAndName("my-oh");
    await user.click(screen.getByTestId("save-agent-profile-btn"));

    await waitFor(() =>
      expect(displayErrorToast).toHaveBeenCalledWith(
        "SETTINGS$AGENT_PROFILE_LLM_REQUIRED",
      ),
    );
    expect(saveMutate).not.toHaveBeenCalled();
  });
});
