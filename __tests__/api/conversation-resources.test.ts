import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  ConversationClient,
  SkillsClient,
} from "@openhands/typescript-client/clients";
import {
  ConversationExecutionStatus,
  type ConversationInfo,
} from "@openhands/typescript-client";
import ConversationResourcesService from "#/api/conversation-resources-service";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import { getActiveBackend } from "#/api/backend-registry/active-store";

const getConversation = vi.spyOn(
  ConversationClient.prototype,
  "getConversation",
);
const listProfiles = vi.spyOn(AgentProfilesService, "listProfiles");
const getSkills = vi.spyOn(SkillsClient.prototype, "getSkills");

function conversation(): ConversationInfo {
  return {
    id: "conversation-id",
    execution_status: ConversationExecutionStatus.IDLE,
    confirmation_policy: { kind: "NeverConfirm" },
    activated_knowledge_skills: [],
    workspace: { working_dir: "/workspace" },
    persistence_dir: "/state",
    launched_agent_profile: {
      agent_profile_id: "profile-id",
      revision: 3,
      secret_refs: ["GITHUB_TOKEN"],
    },
    secret_registry: {
      secret_sources: {
        GITHUB_TOKEN: { kind: "StaticSecret", value: "secret-value" },
      },
    },
    agent: {
      kind: "Agent",
      llm: { model: "launch-model", api_key: "llm-secret" },
      agent_context: {
        skills: [
          {
            name: "project-guide",
            source: "/workspace/.agents/skills/project-guide/SKILL.md",
            content: "private-skill-content",
          },
        ],
      },
      mcp_config: {
        docs: { command: "mcp-docs", env: { TOKEN: "mcp-secret" } },
      },
      condenser: {
        kind: "LLMSummarizingCondenser",
        max_size: 240,
        keep_first: 2,
        llm: { api_key: "condenser-secret" },
      },
      critic: null,
    },
  };
}

describe("conversation resources", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getConversation.mockResolvedValue(conversation());
    listProfiles.mockResolvedValue({
      profiles: [
        {
          id: "profile-id",
          name: "Renamed profile",
          revision: 4,
          agent_kind: "openhands",
          llm_profile_ref: "new-default",
          mcp_server_refs: [],
        },
      ],
      active_agent_profile_id: "different-profile",
    });
  });

  it("reads effective runtime configuration and identifies the launch profile by stable id", async () => {
    const result = await ConversationResourcesService.load(
      "conversation-id",
      getActiveBackend(),
    );
    expect(result.model).toBe("launch-model");
    expect(result.profile).toEqual({
      id: "profile-id",
      name: "Renamed profile",
      launchedRevision: 3,
      currentRevision: 4,
    });
    expect(result.skills?.map((skill) => skill.name)).toEqual([
      "project-guide",
    ]);
    expect(result.mcpServers?.map((server) => server.name)).toEqual(["docs"]);
    expect(result.secretNames).toEqual(["GITHUB_TOKEN"]);
    expect(result.condenser).toEqual({
      kind: "LLMSummarizingCondenser",
      maxSize: 240,
      keepFirst: 2,
    });
  });

  it("projects only displayable values before returning the snapshot to the query cache", async () => {
    const result = await ConversationResourcesService.load(
      "conversation-id",
      getActiveBackend(),
    );
    const serialized = JSON.stringify(result);
    for (const secret of [
      "secret-value",
      "llm-secret",
      "mcp-secret",
      "condenser-secret",
      "private-skill-content",
    ]) {
      expect(serialized).not.toContain(secret);
    }
  });

  it("does not substitute the active profile or defaults for a legacy conversation", async () => {
    const legacy = conversation();
    legacy.launched_agent_profile = null;
    getConversation.mockResolvedValue(legacy);
    const result = await ConversationResourcesService.load(
      "conversation-id",
      getActiveBackend(),
    );
    expect(result.profile).toBeNull();
    expect(result.model).toBe("launch-model");
    expect(listProfiles).not.toHaveBeenCalled();
  });

  it("retains launch provenance and runtime data when the profile was deleted or cannot be read", async () => {
    listProfiles.mockRejectedValue(new Error("unavailable"));
    const result = await ConversationResourcesService.load(
      "conversation-id",
      getActiveBackend(),
    );
    expect(result.profile).toEqual({
      id: "profile-id",
      name: null,
      launchedRevision: 3,
      currentRevision: null,
    });
    expect(result.model).toBe("launch-model");
  });

  it("uses the Cloud conversation runtime URL and session key, without falling back to a local server", async () => {
    const active = getActiveBackend();
    const cloud = {
      ...active,
      backend: { ...active.backend, kind: "cloud" as const },
    };
    await expect(
      ConversationResourcesService.load("conversation-id", cloud),
    ).rejects.toThrow();
    expect(getConversation).not.toHaveBeenCalled();
  });

  it("keeps missing snapshot categories unknown instead of reporting an empty configuration", async () => {
    const snapshot = conversation();
    delete snapshot.agent.agent_context;
    delete snapshot.agent.mcp_config;
    delete snapshot.secret_registry;
    getConversation.mockResolvedValue(snapshot);
    const result = await ConversationResourcesService.load(
      "conversation-id",
      getActiveBackend(),
    );
    expect(result.skills).toBeNull();
    expect(result.mcpServers).toBeNull();
    expect(result.secretNames).toBeNull();
  });

  it("includes lazily discovered project skills using the runtime workspace and launch deny-list", async () => {
    const snapshot = conversation();
    snapshot.workspace = {
      kind: "LocalWorkspace",
      working_dir: "/actual-worktree",
    };
    snapshot.agent.agent_context = {
      skills: [],
      load_project_skills: true,
      disabled_skills: ["disabled-guide"],
    };
    getConversation.mockResolvedValue(snapshot);
    getSkills.mockResolvedValue({
      skills: [
        {
          name: "worktree-guide",
          type: "agentskills",
          source: "/actual-worktree/.agents/skills/worktree-guide/SKILL.md",
          content: "private-worktree-content",
          triggers: [],
        },
        {
          name: "disabled-guide",
          type: "agentskills",
          source: "project",
          content: "disabled",
          triggers: [],
        },
      ],
      sources: {},
    });
    const result = await ConversationResourcesService.load(
      "conversation-id",
      getActiveBackend(),
    );
    expect(getSkills).toHaveBeenCalledWith({
      load_public: false,
      load_user: false,
      load_project: true,
      load_org: false,
      project_dir: "/actual-worktree",
    });
    expect(result.skills?.map((skill) => skill.name)).toEqual([
      "worktree-guide",
    ]);
    expect(JSON.stringify(result)).not.toContain("private-worktree-content");
  });
});
