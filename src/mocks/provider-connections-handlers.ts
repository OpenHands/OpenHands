import { http, HttpResponse } from "msw";

/**
 * In-memory Provider Connection storage for the mock agent-server API.
 * Mirrors the secrets-handlers pattern: a Map keyed by connection id.
 * The key is never echoed back; we only track an `api_key_set` boolean.
 */
interface MockConnection {
  id: string;
  provider: string;
  label?: string;
  models: string[];
  created_at: number;
  last_validated_at: number | null;
  api_key_set: boolean;
  /** Stored only to power validate; never returned. */
  _key?: string;
}

const connections = new Map<string, MockConnection>([
  [
    "conn-1",
    {
      id: "conn-1",
      provider: "openai",
      label: "Work",
      models: ["gpt-4o", "gpt-4o-mini"],
      created_at: 1700000000,
      last_validated_at: 1700000100,
      api_key_set: true,
    },
  ],
]);

let nextId = 2;

function toResponse(c: MockConnection) {
  // Strip the private key field before returning.
  const { _key: _unused, ...rest } = c;
  void _unused;
  return rest;
}

export const PROVIDER_CONNECTIONS_HANDLERS = [
  // GET /api/llm/connections — list
  http.get("*/api/llm/connections", () =>
    HttpResponse.json(Array.from(connections.values()).map(toResponse)),
  ),

  // POST /api/llm/connections — create
  http.post("*/api/llm/connections", async ({ request }) => {
    const body = (await request.json()) as {
      provider?: string;
      key?: string;
      label?: string;
    } | null;
    if (!body?.provider || !body?.key) {
      return HttpResponse.json(
        { detail: "provider and key are required" },
        { status: 400 },
      );
    }
    const id = `conn-${nextId++}`;
    const conn: MockConnection = {
      id,
      provider: body.provider,
      label: body.label,
      models: [],
      created_at: Math.floor(Date.now() / 1000),
      last_validated_at: null,
      api_key_set: true,
      _key: body.key,
    };
    connections.set(id, conn);
    return HttpResponse.json(toResponse(conn), { status: 201 });
  }),

  // GET /api/llm/connections/:id
  http.get("*/api/llm/connections/:id", ({ params }) => {
    const id = String(params.id);
    const conn = connections.get(id);
    if (!conn) {
      return HttpResponse.json(
        { detail: "Connection not found" },
        { status: 404 },
      );
    }
    return HttpResponse.json(toResponse(conn));
  }),

  // PATCH /api/llm/connections/:id
  http.patch("*/api/llm/connections/:id", async ({ request, params }) => {
    const id = String(params.id);
    const conn = connections.get(id);
    if (!conn) {
      return HttpResponse.json(
        { detail: "Connection not found" },
        { status: 404 },
      );
    }
    const body = (await request.json()) as {
      key?: string;
      label?: string;
      models?: string[];
    } | null;
    if (body?.key) {
      conn._key = body.key;
      conn.api_key_set = true;
    }
    if (body?.label !== undefined) conn.label = body.label;
    if (body?.models !== undefined) conn.models = body.models;
    return HttpResponse.json(toResponse(conn));
  }),

  // DELETE /api/llm/connections/:id
  http.delete("*/api/llm/connections/:id", ({ params }) => {
    const id = String(params.id);
    const deleted = connections.delete(id);
    if (!deleted) {
      return HttpResponse.json(
        { detail: "Connection not found" },
        { status: 404 },
      );
    }
    return HttpResponse.json({ id, affected_profiles: [] });
  }),

  // POST /api/llm/connections/:id/validate
  http.post("*/api/llm/connections/:id/validate", ({ params, request }) => {
    const id = String(params.id);
    const conn = connections.get(id);
    if (!conn) {
      return HttpResponse.json(
        { detail: "Connection not found" },
        { status: 404 },
      );
    }
    // A non-empty key validates; surface a small mock catalog for openai.
    const ok = Boolean(conn._key) || conn.api_key_set;
    const models =
      conn.provider === "openai"
        ? ["gpt-4o", "gpt-4o-mini", "o3-mini"]
        : ["mock-model-1"];
    if (ok) {
      conn.models = models;
      conn.last_validated_at = Math.floor(Date.now() / 1000);
    }
    // Only a live probe (?live=true) proves the key; catalog lookups don't.
    const live = new URL(request.url).searchParams.get("live") === "true";
    return HttpResponse.json({
      id,
      provider: conn.provider,
      ok,
      verified: ok && live,
      models: ok ? models : [],
      error: ok ? null : "Invalid API key",
      validated_at: conn.last_validated_at,
    });
  }),

  // POST /api/llm/connections/:id/profiles — create a profile bound to the
  // connection's key by reference.
  http.post(
    "*/api/llm/connections/:id/profiles",
    async ({ params, request }) => {
      const id = String(params.id);
      const conn = connections.get(id);
      if (!conn) {
        return HttpResponse.json(
          { detail: "Connection not found" },
          { status: 404 },
        );
      }
      const body = (await request.json()) as {
        profile_name?: string;
        model?: string;
        base_url?: string;
      } | null;
      if (!body?.profile_name || !body?.model) {
        return HttpResponse.json(
          { detail: "profile_name and model are required" },
          { status: 400 },
        );
      }
      return HttpResponse.json(
        {
          profile_name: body.profile_name,
          model: body.model,
          provider: conn.provider,
          connection_id: id,
        },
        { status: 201 },
      );
    },
  ),
];
