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
  },
  {
    id: "2",
    name: "CI/CD Pipeline",
    prefix: "oh_ci_",
    created_at: "2026-01-15T08:00:00Z",
    last_used_at: null,
  },
];

const apiKeys = new Map<string, ApiKey>(
  DEFAULT_API_KEYS.map((key) => [key.id, key]),
);

export const API_KEYS_HANDLERS = [
  // GET /api/keys - List all API keys
  http.get("/api/keys", () => HttpResponse.json(Array.from(apiKeys.values()))),

  // POST /api/keys - Create a new API key
  http.post("/api/keys", async ({ request }) => {
    const body = (await request.json()) as { name: string };

    if (!body?.name?.trim()) {
      return HttpResponse.json({ error: "Name is required" }, { status: 400 });
    }

    nextId += 1;
    const id = String(nextId);
    const newKey: ApiKey = {
      id,
      name: body.name,
      prefix: `oh_${id}_`,
      created_at: new Date().toISOString(),
      last_used_at: null,
    };
    apiKeys.set(id, newKey);

    const response: CreateApiKeyResponse = {
      id: newKey.id,
      name: newKey.name,
      key: `oh_${id}_sk_mock_${Math.random().toString(36).slice(2, 14)}`,
      prefix: newKey.prefix,
      created_at: newKey.created_at,
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
