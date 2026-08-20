import { describe, expect, it } from "vitest";
import type {
  AgentProfile,
  AgentProfileSaveInput,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import { mergeAgentProfileSaveInput } from "#/components/features/settings/agent-profiles/merge-agent-profile-save-input";

// Mirrors the seeded `default` profile: fields the minimal editor does NOT
// model (condenser, verification, suffix, mcp_server_refs, the disabled_skills
// deny-list) carry non-default values so a wipe-to-defaults would be visible.
// `disabled_skills` rides untyped (the pinned ts-client predates it) — the
// merge must still round-trip it, so the fixture includes it via the cast.
const storedOpenHands = {
  schema_version: 1,
  id: "11111111-1111-1111-1111-111111111111",
  name: "default",
  revision: 3,
  agent_kind: "openhands",
  llm_profile_ref: "old-llm",
  agent: "CodeActAgent",
  system_message_suffix: "Be terse.",
  condenser: { kind: "NoOpCondenserSettings" },
  verification: {
    critic_enabled: true,
    critic_mode: "finish_and_message",
    enable_iterative_refinement: true,
    critic_threshold: 0.8,
    max_refinement_iterations: 5,
    critic_server_url: null,
    critic_model_name: null,
  },
  enable_sub_agents: false,
  enable_switch_llm_tool: false,
  tool_concurrency_limit: 4,
  mcp_server_refs: ["github"],
  disabled_skills: ["deploy-checklist"],
} as unknown as AgentProfile;

const storedAcp = {
  schema_version: 1,
  id: "22222222-2222-2222-2222-222222222222",
  name: "claude",
  revision: 1,
  agent_kind: "acp",
  acp_server: "claude-code",
  acp_model: "claude-opus-4-8",
  acp_session_mode: "bypassPermissions",
  acp_prompt_timeout: 900,
  acp_command: null,
  acp_args: null,
  mcp_server_refs: ["linear"],
} as unknown as AgentProfile;

describe("mergeAgentProfileSaveInput", () => {
  it("preserves unmodeled OpenHands fields under the edited ones", () => {
    const edited: AgentProfileSaveInput = {
      agent_kind: "openhands",
      enable_sub_agents: true,
      llm_profile_ref: "new-llm",
    };

    const merged = mergeAgentProfileSaveInput(storedOpenHands, edited);

    // Edited fields win.
    expect(merged).toMatchObject({
      agent_kind: "openhands",
      enable_sub_agents: true,
      llm_profile_ref: "new-llm",
    });
    // Fields the editor doesn't model survive the whole-profile overwrite.
    expect(merged).toMatchObject({
      condenser: { kind: "NoOpCondenserSettings" },
      verification: { critic_enabled: true, critic_threshold: 0.8 },
      system_message_suffix: "Be terse.",
      mcp_server_refs: ["github"],
      disabled_skills: ["deploy-checklist"],
      enable_switch_llm_tool: false,
      tool_concurrency_limit: 4,
    });
  });

  it("preserves unmodeled ACP fields under the edited ones", () => {
    const edited: AgentProfileSaveInput = {
      agent_kind: "acp",
      acp_server: "claude-code",
      acp_model: "claude-sonnet-5",
      acp_command: null,
      acp_args: null,
    };

    const merged = mergeAgentProfileSaveInput(storedAcp, edited);

    expect(merged).toMatchObject({
      agent_kind: "acp",
      acp_model: "claude-sonnet-5",
      acp_session_mode: "bypassPermissions",
      acp_prompt_timeout: 900,
      mcp_server_refs: ["linear"],
    });
  });

  it("strips server-managed identity from the merge", () => {
    const merged = mergeAgentProfileSaveInput(storedOpenHands, {
      agent_kind: "openhands",
      enable_sub_agents: true,
      llm_profile_ref: "new-llm",
    });

    // The path name is authoritative and the server preserves the namesake's
    // id / bumps revision itself; posting stale identity would only mislead.
    expect(merged).not.toHaveProperty("id");
    expect(merged).not.toHaveProperty("name");
    expect(merged).not.toHaveProperty("revision");
  });

  it("sends a clean variant payload on an agent_kind switch", () => {
    const edited: AgentProfileSaveInput = {
      agent_kind: "acp",
      acp_server: "codex",
      acp_model: null,
      acp_command: null,
      acp_args: null,
    };

    // openhands → acp: carrying stored openhands fields over would produce a
    // mongrel payload the server's extra="forbid" union rejects.
    expect(mergeAgentProfileSaveInput(storedOpenHands, edited)).toEqual(edited);
  });

  it("passes the edited fields through on create (no stored profile)", () => {
    const edited: AgentProfileSaveInput = {
      agent_kind: "openhands",
      enable_sub_agents: false,
      llm_profile_ref: "default",
    };

    expect(mergeAgentProfileSaveInput(null, edited)).toEqual(edited);
  });
});
