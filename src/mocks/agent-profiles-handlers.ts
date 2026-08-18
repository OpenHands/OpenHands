import { http, HttpResponse } from "msw";
import type {
  AgentProfile,
  AgentProfileSummary,
} from "@openhands/typescript-client";

/**
 * In-memory agent-profile store for the mock agent-server API. Keyed by name
 * (agent-server uses name-based lookups, not IDs).
 *
 * Imported as a live module so consumers can seed entries and reset between
 * tests without re-registering handlers.
 */
const profiles = new Map<string, AgentProfile>();

let activeProfileId: string | null = null;

/** Reset the in-memory store (called from test setup, never on main threads). */
export function resetMockAgentProfiles(): void {
  profiles.clear();
  activeProfileId = null;
}

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

/**
 * Mock handlers for the agent-server `/api/agent-profiles` endpoints (the same
 * contract consumed by `AgentProfilesService` and the cloud proxy). Without
 * these, `listProfiles`/`saveProfile` in non-mocked tests hit the real network
 * and reject, which `useCreateConversation` now surfaces as a hard failure
 * (no silent downgrade fallback — see PR #16523).
 *
 * Routes mirror `agent_profiles_router.py` in the agent-server.
 */
export const AGENT_PROFILES_HANDLERS = [
  // GET /api/agent-profiles - List all profiles + the active id.
  http.get("*/api/agent-profiles", async ({ request }) => {
    // Exclude requests carrying a :name segment (handled by the :name route).
    const url = new URL(request.url);
    const pathParts = url.pathname.split("/").filter(Boolean);
    if (pathParts.length > 2) return undefined;

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
    if (!profile) {
      return HttpResponse.json(
        { detail: `Agent profile '${name}' not found` },
        { status: 404 },
      );
    }
    return HttpResponse.json({ name, profile });
  }),

  // POST /api/agent-profiles/:name - Create or overwrite a profile (upsert).
  http.post("*/api/agent-profiles/:name", async ({ params, request }) => {
    const { name } = params;
    if (typeof name !== "string") {
      return HttpResponse.json({ detail: "Invalid name" }, { status: 400 });
    }
    const body = (await request.json()) as AgentProfile;
    profiles.set(name, { ...body, id: body.id ?? name });
    return HttpResponse.json({ name, message: "Agent profile saved." });
  }),

  // DELETE /api/agent-profiles/:name - Delete by name (idempotent).
  http.delete("*/api/agent-profiles/:name", async ({ params }) => {
    const { name } = params;
    if (typeof name !== "string") {
      return HttpResponse.json({ detail: "Invalid name" }, { status: 400 });
    }
    if (activeProfileId === profiles.get(name)?.id) {
      activeProfileId = null;
    }
    profiles.delete(name);
    return HttpResponse.json({ name, message: "Agent profile deleted." });
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
      if (!profile) {
        return HttpResponse.json(
          { detail: `Agent profile '${name}' not found` },
          { status: 404 },
        );
      }
      profiles.delete(name);
      profiles.set(newName, profile);
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
    if (!profile) {
      return HttpResponse.json(
        { detail: `Agent profile '${id}' not found` },
        { status: 404 },
      );
    }
    activeProfileId = id;
    return HttpResponse.json({
      id,
      message: "Agent profile activated.",
      agent_settings_applied: false,
    });
  }),
];
