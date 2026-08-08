import type {
  AgentProfile,
  AgentProfileSaveInput,
  AgentProfileSummary,
} from "@openhands/typescript-client";
import { http, HttpResponse } from "msw";

/**
 * MSW handlers for the `/api/agent-profiles` endpoints.
 *
 * Settings → Agent *is* the agent-profile library (#1571), and the embedded
 * editor behind it is the only surface that renders `AgentSettingsScreen`.
 * Without these handlers the page shows "Failed to load profiles" under
 * `npm run dev:mock`, so the whole agent-settings form is unreachable there.
 *
 * State is per-page-load and deliberately in-memory: create, edit, rename and
 * delete all round-trip so the editor's save path (a whole-profile overwrite)
 * can be exercised, but nothing persists across a refresh.
 */

const DEFAULT_VERIFICATION = {
  critic_enabled: false,
  critic_mode: "off",
  enable_iterative_refinement: false,
  critic_threshold: 0.5,
  max_refinement_iterations: 3,
  critic_server_url: null,
  critic_model_name: null,
};

function makeOpenHandsProfile(
  overrides: Partial<AgentProfile> & { id: string; name: string },
): AgentProfile {
  return {
    schema_version: 1,
    revision: 1,
    mcp_server_refs: null,
    agent_kind: "openhands",
    llm_profile_ref: "default",
    agent: "CodeActAgent",
    skills: [],
    system_message_suffix: null,
    condenser: null,
    verification: DEFAULT_VERIFICATION,
    enable_sub_agents: false,
    tool_concurrency_limit: 1,
    ...overrides,
  } as AgentProfile;
}

const MOCK_AGENT_PROFILES = new Map<string, AgentProfile>();
let activeAgentProfileId: string | null = null;

function seed() {
  MOCK_AGENT_PROFILES.clear();
  for (const profile of [
    makeOpenHandsProfile({
      id: "3f1c1b7e-0000-4000-8000-000000000001",
      name: "default",
    }),
    makeOpenHandsProfile({
      id: "3f1c1b7e-0000-4000-8000-000000000002",
      name: "research",
      enable_sub_agents: true,
      tool_concurrency_limit: 4,
    }),
  ]) {
    MOCK_AGENT_PROFILES.set(profile.name, profile);
  }
  activeAgentProfileId = "3f1c1b7e-0000-4000-8000-000000000001";
}
seed();

function toSummary(profile: AgentProfile): AgentProfileSummary {
  return {
    id: profile.id,
    name: profile.name,
    agent_kind: profile.agent_kind,
    revision: profile.revision,
    llm_profile_ref:
      profile.agent_kind === "openhands" ? profile.llm_profile_ref : null,
    mcp_server_refs: profile.mcp_server_refs,
  };
}

function newProfileId() {
  const suffix = String(MOCK_AGENT_PROFILES.size + 1).padStart(12, "0");
  return `3f1c1b7e-0000-4000-8000-${suffix}`;
}

export const AGENT_PROFILES_HANDLERS = [
  http.get("*/api/agent-profiles", async () =>
    HttpResponse.json({
      profiles: [...MOCK_AGENT_PROFILES.values()].map(toSummary),
      active_agent_profile_id: activeAgentProfileId,
    }),
  ),

  http.get("*/api/agent-profiles/:name", async ({ params }) => {
    const name = String(params.name);
    const profile = MOCK_AGENT_PROFILES.get(name);
    if (!profile) {
      return HttpResponse.json(
        { detail: "Profile not found" },
        { status: 404 },
      );
    }
    return HttpResponse.json({ name, profile });
  }),

  http.post(
    "*/api/agent-profiles/:name/rename",
    async ({ params, request }) => {
      const name = String(params.name);
      const profile = MOCK_AGENT_PROFILES.get(name);
      if (!profile) {
        return HttpResponse.json(
          { detail: "Profile not found" },
          { status: 404 },
        );
      }
      const { new_name: newName } = (await request.json()) as {
        new_name: string;
      };
      if (MOCK_AGENT_PROFILES.has(newName)) {
        return HttpResponse.json(
          { detail: "Name already taken" },
          { status: 409 },
        );
      }
      MOCK_AGENT_PROFILES.delete(name);
      // Rename preserves the stable id and the active pointer with it.
      MOCK_AGENT_PROFILES.set(newName, { ...profile, name: newName });
      return HttpResponse.json({ name: newName, message: "Profile renamed" });
    },
  ),

  http.post("*/api/agent-profiles/:id/activate", async ({ params }) => {
    const id = String(params.id);
    activeAgentProfileId = id;
    return HttpResponse.json({
      id,
      message: "Profile activated",
      agent_settings_applied: false,
    });
  }),

  http.post("*/api/agent-profiles/:name", async ({ params, request }) => {
    const name = String(params.name);
    const input = (await request.json()) as AgentProfileSaveInput;
    const existing = MOCK_AGENT_PROFILES.get(name);
    // Upsert with server-managed identity: the id survives an overwrite and
    // the revision counter moves, matching `save_profile_preserving_identity`.
    MOCK_AGENT_PROFILES.set(name, {
      ...(input as AgentProfile),
      name,
      id: existing?.id ?? newProfileId(),
      revision: (existing?.revision ?? 0) + 1,
    });
    return HttpResponse.json({ name, message: "Profile saved" });
  }),

  http.delete("*/api/agent-profiles/:name", async ({ params }) => {
    const name = String(params.name);
    // Delete is idempotent server-side — a missing name still resolves 200.
    MOCK_AGENT_PROFILES.delete(name);
    return HttpResponse.json({ name, message: "Profile deleted" });
  }),
];
