import { QueryClient } from "@tanstack/react-query";
import { DEFAULT_ENABLED_SKILL_NAMES } from "@openhands/extensions/skills";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  buildDefaultProfileMigration,
  ensureDefaultAgentProfileMigrated,
} from "#/hooks/use-migrate-default-agent-profile";
import { DEFAULT_SETTINGS } from "#/services/settings";
import type { Settings } from "#/types/settings";
import { CATALOG_SKILL_NAMES } from "#/utils/skill-enablement";

vi.mock("#/api/agent-profiles-service/agent-profiles-service.api", () => ({
  __esModule: true,
  default: {
    listProfiles: vi.fn(),
    getProfile: vi.fn(),
    saveProfile: vi.fn(),
  },
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME: "default",
}));
vi.mock("#/api/settings-service/settings-service.api", () => ({
  __esModule: true,
  default: {
    getSettings: vi.fn(),
    saveSettings: vi.fn(),
    invalidateCache: vi.fn(),
  },
}));

const GLOBAL_AGENT = {
  agent_kind: "openhands",
  agent: "CodeActAgent",
  persona: "You review code.",
  tools: [{ name: "terminal", params: {} }],
  tool_concurrency_limit: 3,
  agent_context: { system_message_suffix: "Be terse.", load_memory: true },
  condenser: {
    condenser_kind: "llm_summarizing",
    enabled: false,
    max_size: 80,
  },
  verification: {
    critic_enabled: true,
    critic_mode: "all_actions",
    enable_iterative_refinement: false,
    critic_threshold: 0.7,
    max_refinement_iterations: 2,
    critic_server_url: null,
    critic_model_name: null,
    critic_api_key: "**********",
  },
  enable_classify_and_switch_llm_tool: true,
  active_meta_profile: "router",
};

const settingsWith = (overrides: Partial<Settings> = {}): Settings => ({
  ...DEFAULT_SETTINGS,
  agent_settings: GLOBAL_AGENT,
  enabled_skills: ["github", "docker"],
  disabled_skills: ["my-user-skill"],
  ...overrides,
});

const STORED_DEFAULT = {
  id: "profile-default",
  name: "default",
  revision: 4,
  agent_kind: "openhands",
  llm_profile_ref: "my-llm",
  mcp_server_refs: ["linear"],
  secret_refs: ["GITHUB_TOKEN"],
  persona: null,
  tools: null,
};

describe("buildDefaultProfileMigration", () => {
  it("copies the global behavior the inline launch used, without the critic key", () => {
    expect(buildDefaultProfileMigration(settingsWith())).toEqual({
      public_skills: ["github", "docker"],
      disabled_skills: ["my-user-skill"],
      condenser: GLOBAL_AGENT.condenser,
      verification: {
        critic_enabled: true,
        critic_mode: "all_actions",
        enable_iterative_refinement: false,
        critic_threshold: 0.7,
        max_refinement_iterations: 2,
        critic_server_url: null,
        critic_model_name: null,
      },
      tools: GLOBAL_AGENT.tools,
      persona: "You review code.",
      agent: "CodeActAgent",
      tool_concurrency_limit: 3,
      system_message_suffix: "Be terse.",
      enable_classify_and_switch_llm_tool: true,
      meta_profile_ref: "router",
    });
  });

  it("turns a pre-allow-list deny-list into the skill selection it meant", () => {
    const migrated = buildDefaultProfileMigration(
      settingsWith({
        enabled_skills: undefined,
        disabled_skills: [CATALOG_SKILL_NAMES[0], "my-user-skill"],
      }),
    );

    expect(migrated.public_skills).toEqual(CATALOG_SKILL_NAMES.slice(1));
    expect(migrated.disabled_skills).toEqual(["my-user-skill"]);
  });

  it("gives a fresh workspace the recommended skills", () => {
    const migrated = buildDefaultProfileMigration(
      settingsWith({ enabled_skills: undefined, disabled_skills: [] }),
    );

    expect(migrated.public_skills).toEqual([...DEFAULT_ENABLED_SKILL_NAMES]);
  });

  it("carries only the skills when global settings are left on ACP", () => {
    const migrated = buildDefaultProfileMigration(
      settingsWith({
        agent_settings: { agent_kind: "acp", acp_server: "codex" },
      }),
    );

    expect(migrated).toEqual({
      public_skills: ["github", "docker"],
      disabled_skills: ["my-user-skill"],
    });
  });
});

describe("ensureDefaultAgentProfileMigrated", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    vi.mocked(SettingsService.getSettings).mockResolvedValue(settingsWith());
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: [STORED_DEFAULT],
      active_agent_profile_id: STORED_DEFAULT.id,
    } as never);
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      name: "default",
      profile: STORED_DEFAULT,
    } as never);
  });

  it("merges the global behavior into `default` and records it once", async () => {
    const queryClient = new QueryClient();

    await ensureDefaultAgentProfileMigrated(queryClient, "local-1");
    await ensureDefaultAgentProfileMigrated(queryClient, "local-1");

    expect(AgentProfilesService.saveProfile).toHaveBeenCalledTimes(1);
    const [name, saved] = vi.mocked(AgentProfilesService.saveProfile).mock
      .calls[0];
    expect(name).toBe("default");
    expect(saved).toMatchObject({
      // What the user chose on the profile itself stays.
      llm_profile_ref: "my-llm",
      mcp_server_refs: ["linear"],
      secret_refs: ["GITHUB_TOKEN"],
      // What `default` used to take from global settings moves onto it.
      persona: "You review code.",
      tools: GLOBAL_AGENT.tools,
      public_skills: ["github", "docker"],
      meta_profile_ref: "router",
    });
    expect(saved).not.toHaveProperty("id");
    expect(SettingsService.saveSettings).toHaveBeenCalledWith({
      default_profile_migrated: true,
    });
  });

  it("does nothing once the backend is marked migrated", async () => {
    vi.mocked(SettingsService.getSettings).mockResolvedValue(
      settingsWith({ default_profile_migrated: true }),
    );

    await ensureDefaultAgentProfileMigrated(new QueryClient(), "local-1");

    expect(AgentProfilesService.listProfiles).not.toHaveBeenCalled();
    expect(AgentProfilesService.saveProfile).not.toHaveBeenCalled();
  });

  it("only marks the backend when `default` is an ACP profile", async () => {
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: [{ ...STORED_DEFAULT, agent_kind: "acp" }],
      active_agent_profile_id: STORED_DEFAULT.id,
    } as never);

    await ensureDefaultAgentProfileMigrated(new QueryClient(), "local-1");

    expect(AgentProfilesService.saveProfile).not.toHaveBeenCalled();
    expect(SettingsService.saveSettings).toHaveBeenCalledWith({
      default_profile_migrated: true,
    });
  });

  it("leaves the marker unset when the save fails, so a later call retries", async () => {
    const queryClient = new QueryClient();
    vi.mocked(AgentProfilesService.saveProfile).mockRejectedValueOnce(
      new Error("boom"),
    );

    await expect(
      ensureDefaultAgentProfileMigrated(queryClient, "local-1"),
    ).rejects.toThrow("boom");
    expect(SettingsService.saveSettings).not.toHaveBeenCalled();

    await ensureDefaultAgentProfileMigrated(queryClient, "local-1");
    expect(SettingsService.saveSettings).toHaveBeenCalledTimes(1);
  });
});
