import { http, HttpResponse } from "msw";
import { ApiKey, CreateApiKeyResponse } from "#/api/api-keys";

let nextId = 2;

const DEFAULT_API_KEYS: ApiKey[] = [
  {
    id: "1",
    name: "My Dev Key",
    prefix: "oh_dev_",
    created_at: "2025-12-01T10:00:00Z",
    last_used_at: "2026-02-18T14:30:00Z",
    not_before: null,
    expires_at: null,
    org_id: null,
  },
  {
    id: "2",
    name: "Scheduled Migration",
    prefix: "oh_sched_",
    created_at: "2026-05-01T08:00:00Z",
    last_used_at: null,
    not_before: "2026-07-01T00:00:00Z",
    expires_at: "2026-08-01T00:00:00Z",
    org_id: null,
  },
  {
    id: "3",
    name: "Expired CI Token",
    prefix: "oh_ci_old_",
    created_at: "2025-06-15T08:00:00Z",
    last_used_at: "2025-12-01T09:00:00Z",
    not_before: null,
    expires_at: "2026-01-01T00:00:00Z",
    org_id: null,
  },
];

const apiKeys = new Map<string, ApiKey>(
  DEFAULT_API_KEYS.map((key) => [key.id, key]),
);

interface CreateApiKeyBody {
  name?: string;
  not_before?: string;
  expires_at?: string;
  org_id?: string | null;
}

export const API_KEYS_HANDLERS = [
  // GET /api/keys - List all API keys
  http.get("/api/keys", () => HttpResponse.json(Array.from(apiKeys.values()))),

  // POST /api/keys - Create a new API key
  http.post("/api/keys", async ({ request }) => {
    const body = (await request.json()) as CreateApiKeyBody;

    if (!body?.name?.trim()) {
      return HttpResponse.json({ error: "Name is required" }, { status: 400 });
    }

    if (
      body.not_before &&
      body.expires_at &&
      new Date(body.not_before) >= new Date(body.expires_at)
    ) {
      return HttpResponse.json(
        { error: "not_before must be earlier than expires_at" },
        { status: 400 },
      );
    }

    nextId += 1;
    const id = String(nextId);
    // ``org_id`` is honored verbatim: an explicit ``null`` creates an
    // unbound ("All orgs") key, an UUID string binds the key.
    const orgId = body.org_id ?? null;
    const newKey: ApiKey = {
      id,
      name: body.name,
      prefix: `oh_${id}_`,
      created_at: new Date().toISOString(),
      last_used_at: null,
      not_before: body.not_before ?? null,
      expires_at: body.expires_at ?? null,
      org_id: orgId,
    };
    apiKeys.set(id, newKey);

    const response: CreateApiKeyResponse = {
      id: newKey.id,
      name: newKey.name,
      key: `oh_${id}_sk_mock_${Math.random().toString(36).slice(2, 14)}`,
      prefix: newKey.prefix,
      created_at: newKey.created_at,
      not_before: newKey.not_before,
      expires_at: newKey.expires_at,
      org_id: orgId,
    };

    return HttpResponse.json(response);
  }),

  // DELETE /api/keys/:id - Delete an API key
  http.delete("/api/keys/:id", ({ params }) => {
    const { id } = params;

    if (typeof id === "string" && apiKeys.has(id)) {
      apiKeys.delete(id);
      return HttpResponse.json({ success: true });
    }

    return HttpResponse.json({ error: "Key not found" }, { status: 404 });
  }),

  // GET /api/keys/llm/byor - Get LLM API key
  http.get("/api/keys/llm/byor", () =>
    HttpResponse.json({
      key: "sk-mock-llm-api-key-1234567890abcdef",
    }),
  ),

  // POST /api/keys/llm/byor/refresh - Refresh LLM API key
  http.post("/api/keys/llm/byor/refresh", () =>
    HttpResponse.json({
      key: `sk-mock-llm-refreshed-${Math.random().toString(36).slice(2, 14)}`,
    }),
  ),
];
