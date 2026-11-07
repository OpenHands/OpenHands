import { QueryClient } from "@tanstack/react-query";
import { HttpError } from "@openhands/typescript-client";
import { DEFAULT_ENABLED_SKILL_NAMES } from "@openhands/extensions/skills";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  applyDefaultAgentProfileToGlobal,
  enabledMcpServerRefs,
  sharedDefaultProfileFields,
  llmProfileMirrored,
  mcpServerRemoved,
  mcpServerRenamed,
  routerDetached,
  withSkillSelection,
  updateAgentProfile,
  updateDefaultAgentProfile,
  updateEveryAgentProfile,
} from "#/hooks/mutation/sync-agent-profiles";
import { settleRunningDefaultAgentProfileMigration } from "#/hooks/use-migrate-default-agent-profile";
import type { Settings } from "#/types/settings";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { CATALOG_SKILL_NAMES } from "#/utils/skill-enablement";

vi.mock("#/api/agent-profiles-service/agent-profiles-service.api", () => ({
  __esModule: true,
  default: { listProfiles: vi.fn(), getProfile: vi.fn(), saveProfile: vi.fn() },
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME: "default",
}));
vi.mock("#/api/profiles-service/profiles-service.api", () => ({
  __esModule: true,
  default: { listProfiles: vi.fn(), activateProfile: vi.fn() },
}));
vi.mock("#/api/settings-service/settings-service.api", () => ({
  __esModule: true,
  default: {
    getSettings: vi.fn(),
    saveSettings: vi.fn(),
    patchMcpServer: vi.fn(),
    invalidateCache: vi.fn(),
  },
}));
vi.mock("#/api/backend-registry/active-store", () => ({
  getActiveBackend: vi.fn(),
  isNoBackend: (backend: { id: string }) => backend.id === "no-backend",
}));
vi.mock("#/hooks/use-migrate-default-agent-profile", () => ({
  settleRunningDefaultAgentProfileMigration: vi.fn(),
}));
vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
}));

const OPTIONAL = CATALOG_SKILL_NAMES.find(
  (name) => !DEFAULT_ENABLED_SKILL_NAMES.includes(name),
)!;

const openHands = (name: string, overrides: Record<string, unknown> = {}) => ({
  id: `id-${name}`,
  name,
  revision: 3,
  agent_kind: "openhands",
  llm_profile_ref: "llm",
  mcp_server_refs: null,
  public_skills: [...DEFAULT_ENABLED_SKILL_NAMES],
  disabled_skills: [],
  ...overrides,
});

const useBackend = (kind: "local" | "cloud") =>
  vi.mocked(getActiveBackend).mockReturnValue({
    backend: { id: "local-1", kind },
  } as never);

const storeProfiles = (...profiles: ReturnType<typeof openHands>[]) => {
  vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
    profiles,
  } as never);
  vi.mocked(AgentProfilesService.getProfile).mockImplementation(
    async (name) => {
      const profile = profiles.find((entry) => entry.name === name);
      if (!profile) throw new HttpError(404, "Not Found", {}, "missing");
      return { name, profile } as never;
    },
  );
};

const MCP_SERVERS = {
  github: { transport: "http", url: "https://github.example/mcp" },
  slack: { transport: "stdio", command: "npx", enabled: false },
};

const settingsWith = (overrides: Partial<Settings>): Settings =>
  ({
    enabled_skills: [...DEFAULT_ENABLED_SKILL_NAMES],
    disabled_skills: [],
    mcp_config: MCP_SERVERS,
    ...overrides,
  }) as unknown as Settings;

describe("profile changes", () => {
  it("renames and drops an MCP server only where a profile lists it", () => {
    const restricted = openHands("p", {
      mcp_server_refs: ["github", "slack"],
    }) as never;
    expect(mcpServerRenamed("github", "gh")(restricted)).toEqual({
      mcp_server_refs: ["gh", "slack"],
    });
    expect(mcpServerRemoved("github")(restricted)).toEqual({
      mcp_server_refs: ["slack"],
    });
    expect(mcpServerRemoved("linear")(restricted)).toBeNull();
    expect(mcpServerRemoved("github")(openHands("all") as never)).toBeNull();
  });

  it("sets default's whole skill list", () => {
    const selection = { enabledSkills: [OPTIONAL], disabledSkills: ["mine"] };
    expect(
      withSkillSelection(selection)(openHands("default") as never),
    ).toEqual({
      public_skills: [OPTIONAL],
      disabled_skills: ["mine"],
    });
    expect(
      withSkillSelection(selection)(
        openHands("default", {
          public_skills: [OPTIONAL],
          disabled_skills: ["mine"],
        }) as never,
      ),
    ).toBeNull();
    expect(
      withSkillSelection(selection)({
        ...openHands("default"),
        agent_kind: "acp",
      } as never),
    ).toBeNull();
  });

  it("points default at the active LLM profile", () => {
    expect(llmProfileMirrored("fast")(openHands("default") as never)).toEqual({
      llm_profile_ref: "fast",
    });
    expect(llmProfileMirrored("llm")(openHands("default") as never)).toBeNull();
  });

  it("reads the servers switched on, or null when all are", () => {
    expect(enabledMcpServerRefs(settingsWith({}))).toEqual(["github"]);
    expect(
      enabledMcpServerRefs(
        settingsWith({
          mcp_config: { github: { transport: "http", url: "u" } },
        } as never),
      ),
    ).toBeNull();
  });
});

describe("profile writes", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    vi.clearAllMocks();
    queryClient = new QueryClient();
    useBackend("local");
    vi.mocked(settleRunningDefaultAgentProfileMigration).mockResolvedValue(
      undefined,
    );
    vi.mocked(AgentProfilesService.saveProfile).mockResolvedValue({} as never);
  });

  it("saves a change over the stored default after a running migration", async () => {
    storeProfiles(openHands("default"));

    await updateDefaultAgentProfile(queryClient, llmProfileMirrored("fast"));

    expect(settleRunningDefaultAgentProfileMigration).toHaveBeenCalledWith(
      queryClient,
      "local-1",
    );
    const [name, saved] = vi.mocked(AgentProfilesService.saveProfile).mock
      .calls[0];
    expect(name).toBe("default");
    expect(saved).toMatchObject({
      agent_kind: "openhands",
      llm_profile_ref: "fast",
      public_skills: [...DEFAULT_ENABLED_SKILL_NAMES],
    });
    expect(saved).not.toHaveProperty("revision");
  });

  it("updates every profile that lists a removed MCP server", async () => {
    storeProfiles(
      openHands("default"),
      openHands("uses-it", { mcp_server_refs: ["github", "slack"] }),
      openHands("other", { mcp_server_refs: ["slack"] }),
    );

    await updateEveryAgentProfile(queryClient, mcpServerRemoved("github"));

    expect(AgentProfilesService.saveProfile).toHaveBeenCalledTimes(1);
    expect(AgentProfilesService.saveProfile).toHaveBeenCalledWith(
      "uses-it",
      expect.objectContaining({ mcp_server_refs: ["slack"] }),
    );
  });

  it("does nothing on Cloud, without a default, or when nothing changes", async () => {
    useBackend("cloud");
    await updateDefaultAgentProfile(queryClient, llmProfileMirrored("fast"));
    expect(AgentProfilesService.getProfile).not.toHaveBeenCalled();

    useBackend("local");
    storeProfiles();
    await updateDefaultAgentProfile(queryClient, llmProfileMirrored("fast"));
    storeProfiles(openHands("default"));
    await updateDefaultAgentProfile(queryClient, llmProfileMirrored("llm"));

    expect(AgentProfilesService.saveProfile).not.toHaveBeenCalled();
    expect(displayErrorToast).not.toHaveBeenCalled();
  });

  it("saves one profile's change on Cloud, without the local migration", async () => {
    useBackend("cloud");
    storeProfiles(openHands("reviewer"));

    await updateAgentProfile(
      queryClient,
      "reviewer",
      llmProfileMirrored("fast"),
    );

    expect(settleRunningDefaultAgentProfileMigration).not.toHaveBeenCalled();
    expect(AgentProfilesService.saveProfile).toHaveBeenCalledWith(
      "reviewer",
      expect.objectContaining({ llm_profile_ref: "fast" }),
    );
  });

  it("runs writes one at a time, so the second reads the first's save", async () => {
    let stored = openHands("default", { mcp_server_refs: ["a", "b", "c"] });
    vi.mocked(AgentProfilesService.getProfile).mockImplementation(
      async () => ({ name: "default", profile: stored }) as never,
    );
    vi.mocked(AgentProfilesService.saveProfile).mockImplementation(
      async (_name, profile) => {
        await new Promise((resolve) => {
          setTimeout(resolve, 5);
        });
        stored = { ...stored, ...(profile as typeof stored) };
        return {} as never;
      },
    );

    await Promise.all([
      updateDefaultAgentProfile(queryClient, mcpServerRemoved("a")),
      updateDefaultAgentProfile(queryClient, mcpServerRemoved("b")),
    ]);

    expect(stored.mcp_server_refs).toEqual(["c"]);
  });

  it("toasts a failed save and keeps later writes running", async () => {
    storeProfiles(openHands("default"));
    vi.mocked(AgentProfilesService.saveProfile).mockRejectedValueOnce(
      new Error("boom"),
    );
    vi.spyOn(console, "error").mockImplementation(() => undefined);

    await updateDefaultAgentProfile(queryClient, llmProfileMirrored("fast"));
    await updateDefaultAgentProfile(queryClient, llmProfileMirrored("slow"));

    expect(displayErrorToast).toHaveBeenCalledTimes(1);
    expect(AgentProfilesService.saveProfile).toHaveBeenCalledTimes(2);
  });

  it("detaches a deleted router, and can reject so the delete stops", async () => {
    storeProfiles(
      openHands("default"),
      openHands("routed", {
        meta_profile_ref: "fast",
        enable_classify_and_switch_llm_tool: true,
      }),
    );

    await updateEveryAgentProfile(queryClient, routerDetached("fast"), {
      throwOnError: true,
    });

    expect(AgentProfilesService.saveProfile).toHaveBeenCalledTimes(1);
    expect(AgentProfilesService.saveProfile).toHaveBeenCalledWith(
      "routed",
      expect.objectContaining({
        meta_profile_ref: null,
        enable_classify_and_switch_llm_tool: false,
      }),
    );

    vi.mocked(AgentProfilesService.saveProfile).mockRejectedValueOnce(
      new Error("boom"),
    );
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    await expect(
      updateEveryAgentProfile(queryClient, routerDetached("fast"), {
        throwOnError: true,
      }),
    ).rejects.toThrow("boom");
    expect(displayErrorToast).not.toHaveBeenCalled();
  });

  it("lets the caller report a failure instead of the default toast", async () => {
    storeProfiles(openHands("default"));
    vi.mocked(AgentProfilesService.saveProfile).mockRejectedValueOnce(
      new Error("boom"),
    );
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const onError = vi.fn();

    await updateDefaultAgentProfile(queryClient, llmProfileMirrored("fast"), {
      onError,
    });

    expect(onError).toHaveBeenCalledWith(new Error("boom"));
    expect(displayErrorToast).not.toHaveBeenCalled();
  });
});

describe("applyDefaultAgentProfileToGlobal", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    vi.clearAllMocks();
    queryClient = new QueryClient();
    // As the server sends it: servers under `agent_settings`, a stale flat map.
    vi.mocked(SettingsService.getSettings).mockResolvedValue(
      settingsWith({
        mcp_config: {},
        agent_settings: { mcp_config: MCP_SERVERS },
      } as never),
    );
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: [],
      active_profile: "llm",
    } as never);
  });

  it("writes default's LLM and MCP servers to the global pages, not its skills", async () => {
    const input = {
      agent_kind: "openhands",
      llm_profile_ref: "fast",
      public_skills: [OPTIONAL],
      disabled_skills: ["mine"],
      mcp_server_refs: ["slack"],
      secret_refs: ["GITHUB_TOKEN"],
    } as never;
    await applyDefaultAgentProfileToGlobal(queryClient, input);
    const saved = sharedDefaultProfileFields(input);

    expect(SettingsService.saveSettings).not.toHaveBeenCalled();
    expect(ProfilesService.activateProfile).toHaveBeenCalledWith("fast");
    expect(SettingsService.patchMcpServer).toHaveBeenCalledWith("github", {
      enabled: false,
    });
    expect(SettingsService.patchMcpServer).toHaveBeenCalledWith("slack", {
      enabled: true,
    });
    expect(saved).toMatchObject({ mcp_server_refs: null, secret_refs: null });
  });

  it("writes nothing when default already matches the global pages", async () => {
    const input = {
      agent_kind: "openhands",
      llm_profile_ref: "llm",
      public_skills: [...DEFAULT_ENABLED_SKILL_NAMES],
      disabled_skills: [],
      mcp_server_refs: ["github"],
    } as never;
    await applyDefaultAgentProfileToGlobal(queryClient, input);
    const saved = sharedDefaultProfileFields(input);

    expect(SettingsService.saveSettings).not.toHaveBeenCalled();
    expect(ProfilesService.activateProfile).not.toHaveBeenCalled();
    expect(SettingsService.patchMcpServer).not.toHaveBeenCalled();
    expect(saved).toMatchObject({ mcp_server_refs: null });
    expect(saved).not.toHaveProperty("secret_refs");
  });

  it("switches every server on when default takes all of them", async () => {
    await applyDefaultAgentProfileToGlobal(queryClient, {
      agent_kind: "acp",
      mcp_server_refs: null,
    } as never);

    expect(SettingsService.patchMcpServer).toHaveBeenCalledTimes(1);
    expect(SettingsService.patchMcpServer).toHaveBeenCalledWith("slack", {
      enabled: true,
    });
    expect(SettingsService.saveSettings).not.toHaveBeenCalled();
  });
});
