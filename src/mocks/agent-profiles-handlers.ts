import { http, HttpResponse } from "msw";
import type {
  AgentProfile,
  AgentProfileSaveInput,
  AgentProfileSummary,
} from "@openhands/typescript-client";

/**
 * In-memory agent-profile store for the mock agent-server API. Keyed by name
 * (agent-server uses name-based lookups, not IDs).
 *
 * Two consumers depend on it:
 *
 * - Settings → Agent *is* the agent-profile library (#1571), and the embedded
 *   editor behind it is the only surface that renders `AgentSettingsScreen`.
 *   Without a seeded store the page shows "Failed to load profiles" under
 *   `npm run dev:mock`, so the whole agent-settings form is unreachable there.
 * - `useCreateConversation` no longer downgrades silently when the profile
 *   list fails (#16523): in non-mocked tests `listProfiles` would hit the real
 *   network, reject, and surface as a hard failure.
 *
 * State is per-module-load and deliberately in-memory: create, edit, rename
 * and delete all round-trip so the editor's save path (a whole-profile
 * overwrite) can be exercised, but nothing persists across a refresh.
 *
 * Imported as a live module so consumers can seed entries and reset between
 * tests without re-registering handlers.
 */
const profiles = new Map<string, AgentProfile>();

let activeProfileId: string | null = null;

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

/** Stable id of the seeded `default` profile (the active one after a seed). */
export const MOCK_DEFAULT_AGENT_PROFILE_ID =
  "3f1c1b7e-0000-4000-8000-000000000001";

const SEEDED_PROFILES: readonly AgentProfile[] = [
  makeOpenHandsProfile({
    id: MOCK_DEFAULT_AGENT_PROFILE_ID,
    name: "default",
  }),
  makeOpenHandsProfile({
    id: "3f1c1b7e-0000-4000-8000-000000000002",
    name: "research",
    enable_sub_agents: true,
    tool_concurrency_limit: 4,
  }),
];

// Monotonic so ids minted after a delete never collide with a live profile.
let nextProfileSeq = SEEDED_PROFILES.length + 1;

function newProfileId(): string {
  const suffix = String(nextProfileSeq).padStart(12, "0");
  nextProfileSeq += 1;
  return `3f1c1b7e-0000-4000-8000-${suffix}`;
}

/** Reset the in-memory store to empty (no profiles, no active pointer). */
export function resetMockAgentProfiles(): void {
  profiles.clear();
  activeProfileId = null;
  nextProfileSeq = SEEDED_PROFILES.length + 1;
}

/**
 * Restore the seeded store: a `default` profile (active, mirroring what the
 * agent-server seeds on first run) plus a `research` profile so the library
 * has something to switch between.
 */
export function seedMockAgentProfiles(): void {
  resetMockAgentProfiles();
  for (const profile of SEEDED_PROFILES) {
    profiles.set(profile.name, { ...profile });
  }
  activeProfileId = MOCK_DEFAULT_AGENT_PROFILE_ID;
}
seedMockAgentProfiles();

function toSummary(name: string, profile: AgentProfile): AgentProfileSummary {
  return {
    id: profile.id,
    name,
    agent_kind: profile.agent_kind,
    revision: profile.revision,
    llm_profile_ref:
      profile.agent_kind === "openhands" ? profile.llm_profile_ref : null,
    mcp_server_refs: profile.mcp_server_refs,
  };
}

function notFound(name: string) {
  return HttpResponse.json(
    { detail: `Agent profile '${name}' not found` },
    { status: 404 },
  );
}

/**
 * Mock handlers for the agent-server `/api/agent-profiles` endpoints (the same
 * contract consumed by `AgentProfilesService` and the cloud proxy).
 *
 * Routes mirror `agent_profiles_router.py` in the agent-server. MSW anchors its
 * path matcher, so the list route never claims the `:name` routes below and no
 * extra path guarding is needed.
 */
export const AGENT_PROFILES_HANDLERS = [
  // GET /api/agent-profiles - List all profiles + the active id.
  http.get("*/api/agent-profiles", async () => {
    const summaries = Array.from(profiles.entries()).map(([name, profile]) =>
      toSummary(name, profile),
    );
    return HttpResponse.json({
      profiles: summaries,
      active_agent_profile_id: activeProfileId,
    });
  }),

  // GET /api/agent-profiles/:name - Fetch a single profile.
  http.get("*/api/agent-profiles/:name", async ({ params }) => {
    const { name } = params;
    if (typeof name !== "string") {
      return HttpResponse.json({ detail: "Invalid name" }, { status: 400 });
    }
    const profile = profiles.get(name);
    if (!profile) return notFound(name);
    return HttpResponse.json({ name, profile });
  }),

  // POST /api/agent-profiles/:name/rename - Rename a profile.
  http.post(
    "*/api/agent-profiles/:name/rename",
    async ({ params, request }) => {
      const { name } = params;
      if (typeof name !== "string") {
        return HttpResponse.json({ detail: "Invalid name" }, { status: 400 });
      }
      const body = (await request.json()) as { new_name?: string } | null;
      const newName = body?.new_name;
      if (!newName) {
        return HttpResponse.json(
          { detail: "new_name is required" },
          { status: 422 },
        );
      }
      const profile = profiles.get(name);
      if (!profile) return notFound(name);
      if (newName !== name && profiles.has(newName)) {
        return HttpResponse.json(
          { detail: `Agent profile '${newName}' already exists` },
          { status: 409 },
        );
      }
      profiles.delete(name);
      // Rename preserves the stable id, and the active pointer with it.
      profiles.set(newName, { ...profile, name: newName });
      return HttpResponse.json({
        name: newName,
        message: "Agent profile renamed.",
      });
    },
  ),

  // POST /api/agent-profiles/:id/activate - Activate by stable UUID (pointer-only).
  http.post("*/api/agent-profiles/:id/activate", async ({ params }) => {
    const { id } = params;
    if (typeof id !== "string") {
      return HttpResponse.json({ detail: "Invalid id" }, { status: 400 });
    }
    const profile = Array.from(profiles.values()).find((p) => p.id === id);
    if (!profile) return notFound(id);
    activeProfileId = id;
    return HttpResponse.json({
      id,
      message: "Agent profile activated.",
      agent_settings_applied: false,
    });
  }),

  // POST /api/agent-profiles/:name - Create or overwrite a profile (upsert).
  http.post("*/api/agent-profiles/:name", async ({ params, request }) => {
    const { name } = params;
    if (typeof name !== "string") {
      return HttpResponse.json({ detail: "Invalid name" }, { status: 400 });
    }
    const body = (await request.json()) as AgentProfileSaveInput;
    const existing = profiles.get(name);
    // Upsert with server-managed identity: the id survives an overwrite and
    // the revision counter moves, matching `save_profile_preserving_identity`.
    // A caller-supplied id is honoured only on create, so tests can seed a
    // profile with a known id and activate it.
    profiles.set(name, {
      ...(body as AgentProfile),
      name,
      id: existing?.id ?? body.id ?? newProfileId(),
      revision: (existing?.revision ?? 0) + 1,
    });
    return HttpResponse.json({ name, message: "Agent profile saved." });
  }),

  // DELETE /api/agent-profiles/:name - Delete by name (idempotent).
  http.delete("*/api/agent-profiles/:name", async ({ params }) => {
    const { name } = params;
    if (typeof name !== "string") {
      return HttpResponse.json({ detail: "Invalid name" }, { status: 400 });
    }
    // Deleting the active profile clears the pointer rather than leaving it
    // dangling; a missing name still resolves 200, as on the server.
    if (activeProfileId === profiles.get(name)?.id) {
      activeProfileId = null;
    }
    profiles.delete(name);
    return HttpResponse.json({ name, message: "Agent profile deleted." });
  }),
];
