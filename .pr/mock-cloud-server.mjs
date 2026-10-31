#!/usr/bin/env node
// Minimal stateful mock of the OpenHands Cloud app-server surface the Agent
// Canvas talks to directly (see src/api/cloud/*). Evidence tooling for
// PR #18913 (#18883): the only scripted part is the cloud app-server — the
// Canvas build, its agent-kind sync and the UI are the real product.
//
// State is seeded as an account right after an ACP onboarding:
//   agent_settings.agent_kind = "acp" (as onboarding's
//   buildAcpAgentSettingsDiff writes it), the ACP profile "codex" active,
//   and the OpenHands profile "default" available but not active.
// Every request is appended to a JSONL log so a run can show exactly which
// settings writes happened — the sync under test writes agent_kind through
// POST /api/v1/settings.
//
// Run: node .pr/mock-cloud-server.mjs   (env: MOCK_CLOUD_PORT, MOCK_CLOUD_LOG)

import { createServer } from "node:http";
import { appendFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";

const PORT = Number(process.env.MOCK_CLOUD_PORT ?? 18901);
const TOKEN = "mock-cloud-key";
const ORG_ID = "org-personal-1";
const USER_ID = ORG_ID; // personal workspace contract: org_id === user_id
const LOG_FILE =
  process.env.MOCK_CLOUD_LOG ?? join(".pr", "cloud-requests.jsonl");
mkdirSync(dirname(LOG_FILE), { recursive: true });

const log = (entry) => {
  appendFileSync(LOG_FILE, `${JSON.stringify({ ts: new Date().toISOString(), ...entry })}\n`);
};

// ── Seed: the account state right after an ACP onboarding ──────────────
const settings = {
  language: "en",
  user_consents_to_analytics: true,
  llm_model: "mock-llm-model",
  llm_base_url: "",
  agent: "CodeActAgent",
  llm_api_key_set: true,
  llm_api_key_is_set: true,
  search_api_key_set: false,
  confirmation_mode: false,
  security_analyzer: null,
  max_iterations: 500,
  remote_runtime_resource_factor: null,
  provider_tokens_set: {},
  enable_default_condenser: true,
  condenser_max_size: null,
  enable_sound_notifications: false,
  enable_proactive_conversation_starters: false,
  enable_solvability_analysis: false,
  provider_tokens: {},
  provider_tokens_env: {},
  working_dir: "/workspace",
  // The stale kind under test: written by ACP onboarding, never reset by
  // the profile flow before this fix.
  agent_settings: {
    agent_kind: "acp",
    acp_server: "codex",
    acp_model: "gpt-5.2",
    acp_command: [],
    acp_args: [],
    llm: {
      model: "mock-llm-model",
      base_url: "",
      api_key: null,
      temperature: null,
    },
    condenser: { enable_default_condenser: true, condenser_max_size: null },
    verification: { critic_enabled: false, enable_iterative_refinement: false },
    tool_concurrency_limit: 1,
  },
  conversation_settings: {
    max_iterations: 500,
    confirmation_mode: false,
    security_analyzer: null,
  },
  misc_settings: {
    app_preferences: {
      language: "en",
      user_consents_to_analytics: true,
      enable_sound_notifications: false,
      git_user_name: "Mock User",
      git_user_email: "mock@example.com",
      title_llm_profile: null,
      disabled_skills: [],
    },
  },
};

const profiles = new Map([
  [
    "default",
    {
      agent_kind: "openhands",
      id: "prof-default",
      name: "default",
      revision: 1,
      mcp_server_refs: null,
      llm_profile_ref: "mock-llm",
    },
  ],
  [
    "codex",
    {
      agent_kind: "acp",
      id: "prof-codex",
      name: "codex",
      revision: 1,
      mcp_server_refs: null,
      acp_server: "codex",
      acp_model: "gpt-5.2",
      acp_session_mode: null,
      acp_prompt_timeout: 60,
      acp_command: null,
      acp_args: null,
    },
  ],
]);
let activeProfileId = "prof-codex"; // ACP onboarding left the ACP profile active

let conversationSeq = 0;
const conversations = new Map();

// ── Helpers ────────────────────────────────────────────────────────────
const json = (res, status, body) => {
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,POST,PATCH,DELETE,OPTIONS",
    "Access-Control-Allow-Headers": "Authorization,Content-Type,X-Org-Id,X-Expose-Secrets,X-Session-API-Key",
  });
  res.end(JSON.stringify(body));
};

const readBody = (req) =>
  new Promise((resolve) => {
    let raw = "";
    req.on("data", (chunk) => (raw += chunk));
    req.on("end", () => {
      try {
        resolve(raw ? JSON.parse(raw) : {});
      } catch {
        resolve({});
      }
    });
  });

// One-level merge matching how the app reads agent_settings back.
function mergeDiff(target, diff) {
  for (const [key, value] of Object.entries(diff)) {
    if (
      value !== null &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      target[key] !== null &&
      typeof target[key] === "object" &&
      !Array.isArray(target[key])
    ) {
      target[key] = { ...target[key], ...value };
    } else {
      target[key] = value;
    }
  }
}

// ── Server ─────────────────────────────────────────────────────────────
const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://127.0.0.1:${PORT}`);
  const path = url.pathname;
  const body =
    req.method === "POST" || req.method === "PATCH" ? await readBody(req) : null;

  if (req.method === "OPTIONS") {
    return json(res, 204, {});
  }

  const auth = req.headers.authorization ?? "";
  if (auth !== `Bearer ${TOKEN}`) {
    log({ method: req.method, path, status: 401 });
    return json(res, 401, { detail: "Invalid or missing API key." });
  }

  log({ method: req.method, path, body: body ?? undefined });

  // Organizations + identity
  if (path === "/api/organizations") {
    return json(res, 200, {
      items: [{ id: ORG_ID, name: "Personal Workspace", is_visible: true }],
      current_org_id: ORG_ID,
    });
  }
  if (path === "/api/keys/current") {
    return json(res, 200, { org_id: ORG_ID, name: "mock", status: "active" });
  }
  if (path === `/api/organizations/${ORG_ID}/me`) {
    return json(res, 200, {
      org_id: ORG_ID,
      user_id: USER_ID,
      role: "owner",
      // No `permissions` field: the GUI falls back to the role (owner can
      // manage profiles). A present empty array would mean "no permissions".
    });
  }

  // Settings (the endpoint the sync under test writes)
  if (path === "/api/v1/settings" && req.method === "GET") {
    return json(res, 200, settings);
  }
  if (path === "/api/v1/settings" && req.method === "POST") {
    if (body.agent_settings_diff) {
      mergeDiff(settings.agent_settings, body.agent_settings_diff);
    }
    if (body.conversation_settings_diff) {
      mergeDiff(
        settings.conversation_settings,
        body.conversation_settings_diff,
      );
    }
    for (const [key, value] of Object.entries(body)) {
      if (key !== "agent_settings_diff" && key !== "conversation_settings_diff" && value !== undefined) {
        settings[key] = value;
      }
    }
    return json(res, 200, settings);
  }
  if (path === "/api/v1/settings/agent-schema") {
    return json(res, 200, { properties: {} });
  }
  if (path === "/api/v1/settings/conversation-schema") {
    return json(res, 200, { properties: {} });
  }

  // LLM profiles (org route; no org role enforced in the mock)
  if (path === `/api/organizations/${ORG_ID}/profiles` && req.method === "GET") {
    return json(res, 200, {
      profiles: [
        { name: "mock-llm", model: "mock-llm-model", base_url: null, api_key_set: true },
      ],
      active_profile: "mock-llm",
    });
  }

  // Agent profiles (same paths as the local agent-server contract)
  if (path === "/api/agent-profiles" && req.method === "GET") {
    return json(res, 200, {
      profiles: [...profiles.values()].map((p) => ({
        id: p.id,
        name: p.name,
        agent_kind: p.agent_kind,
        revision: p.revision,
        llm_profile_ref: p.llm_profile_ref ?? null,
        mcp_server_refs: p.mcp_server_refs ?? null,
      })),
      active_agent_profile_id: activeProfileId,
    });
  }
  let match = path.match(/^\/api\/agent-profiles\/([^/]+)\/activate$/);
  if (match && req.method === "POST") {
    const profileId = decodeURIComponent(match[1]);
    if (![...profiles.values()].some((p) => p.id === profileId)) {
      return json(res, 404, { detail: "Profile not found." });
    }
    activeProfileId = profileId;
    log({ event: "activated", profileId, agent_kind: [...profiles.values()].find((p) => p.id === profileId).agent_kind });
    return json(res, 200, {
      id: profileId,
      message: "Profile activated",
      agent_settings_applied: false,
    });
  }
  match = path.match(/^\/api\/agent-profiles\/([^/]+)$/);
  if (match && req.method === "GET") {
    const profile = profiles.get(decodeURIComponent(match[1]));
    return profile
      ? json(res, 200, { name: profile.name, profile })
      : json(res, 404, { detail: "Profile not found." });
  }
  if (match && req.method === "POST") {
    const name = decodeURIComponent(match[1]);
    const existing = profiles.get(name);
    const profile = {
      ...(existing ?? {}),
      ...body,
      id: existing?.id ?? `prof-${name.toLowerCase()}`,
      name,
      revision: (existing?.revision ?? 0) + 1,
    };
    profiles.set(name, profile);
    return json(res, 200, { name, message: "Profile saved" });
  }
  if (match && req.method === "DELETE") {
    const name = decodeURIComponent(match[1]);
    const profile = profiles.get(name);
    if (!profile) return json(res, 404, { detail: "Profile not found." });
    profiles.delete(name);
    if (activeProfileId === profile.id) activeProfileId = null;
    return json(res, 200, { name, message: "Profile deleted" });
  }

  // Conversations (search/list/create; enough for the home page and a
  // scripted launch — the launch composes from the settings kind recorded
  // above, which the log shows).
  if (path === "/api/v1/app-conversations/search" && req.method === "GET") {
    return json(res, 200, { items: [], next_page_id: null });
  }
  if (path === "/api/v1/app-conversations" && req.method === "GET") {
    return json(res, 200, { items: [...conversations.values()], next_page_id: null });
  }
  if (path === "/api/v1/app-conversations" && req.method === "POST") {
    conversationSeq += 1;
    const conversationId = `conv-${String(conversationSeq).padStart(3, "0")}`;
    conversations.set(conversationId, {
      conversation_id: conversationId,
      ...body,
      created_at: new Date().toISOString(),
      status: "starting",
      session_api_key: "mock-session-key",
    });
    log({
      event: "conversation_created",
      conversation_id: conversationId,
      // What the launch would compose from: the recorded settings kind.
      settings_agent_kind_at_launch: settings.agent_settings?.agent_kind,
      settings_agent_profile_id: body?.agent_profile_id ?? null,
    });
    return json(res, 200, {
      conversation_id: conversationId,
      status: "starting",
      session_api_key: "mock-session-key",
      created_at: new Date().toISOString(),
    });
  }

  // Web client config + generic fallback
  if (path === "/api/v1/web-client/config") {
    return json(res, 200, {});
  }
  if (path === "/api/v1/config/models/search") {
    return json(res, 200, { items: [], next_page_id: null });
  }
  if (path === "/api/v1/config/providers/search") {
    return json(res, 200, { items: [], next_page_id: null });
  }
  log({ method: req.method, path, status: 404, note: "unhandled" });
  return json(res, 404, { detail: `Mock cloud has no handler for ${req.method} ${path}` });
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`[mock-cloud] listening on http://127.0.0.1:${PORT}`);
  console.log(`[mock-cloud] token: ${TOKEN}`);
  console.log(`[mock-cloud] log: ${LOG_FILE}`);
});
