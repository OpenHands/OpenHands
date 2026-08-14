import { http, HttpResponse } from "msw";

/**
 * In-memory Model Provider storage for the mock agent-server API
 * (OpenHands/OpenHands#15492). Each provider owns its models (nested,
 * editable) and holds one API key. The key is never echoed back; we only
 * track an `api_key_set` boolean.
 */
interface MockModel {
  name: string;
  wire_api: "auto" | "chat" | "responses" | null;
}

interface MockProvider {
  id: string;
  kind: string;
  display_name: string;
  base_url?: string;
  wire_api: "auto" | "chat" | "responses";
  custom_headers: Record<string, string>;
  models: MockModel[];
  created_at: number;
  updated_at: number;
  api_key_set: boolean;
  /** Stored only to power the test probe; never returned. */
  _key?: string;
}

const providers = new Map<string, MockProvider>([
  [
    "prov-openai",
    {
      id: "prov-openai",
      kind: "openai",
      display_name: "OpenAI",
      base_url: "https://api.openai.com/v1",
      wire_api: "auto",
      custom_headers: {},
      models: [
        { name: "gpt-5.6-luna", wire_api: null },
        { name: "gpt-5.6-sol", wire_api: null },
        { name: "gpt-5.6-terra", wire_api: null },
      ],
      created_at: 1700000000,
      updated_at: 1700000100,
      api_key_set: true,
      _key: "sk-mock",
    },
  ],
]);

let nextId = 1;

function toResponse(p: MockProvider) {
  const { _key: _unused, ...rest } = p;
  void _unused;
  return rest;
}

const now = () => Math.floor(Date.now() / 1000);

export const MODEL_PROVIDERS_HANDLERS = [
  // GET /api/llm/model-providers — list
  http.get("*/api/llm/model-providers", () =>
    HttpResponse.json(Array.from(providers.values()).map(toResponse)),
  ),

  // POST /api/llm/model-providers — create
  http.post("*/api/llm/model-providers", async ({ request }) => {
    const body = (await request.json()) as {
      kind?: string;
      display_name?: string;
      key?: string;
      base_url?: string;
      wire_api?: "auto" | "chat" | "responses";
      custom_headers?: Record<string, string>;
      models?: MockModel[];
    } | null;
    if (!body?.kind || !body?.display_name) {
      return HttpResponse.json(
        { detail: "kind and display_name are required" },
        { status: 400 },
      );
    }
    const id = `prov-${nextId++}`;
    const provider: MockProvider = {
      id,
      kind: body.kind,
      display_name: body.display_name,
      base_url: body.base_url,
      wire_api: body.wire_api ?? "auto",
      custom_headers: body.custom_headers ?? {},
      models: body.models ?? [],
      created_at: now(),
      updated_at: now(),
      api_key_set: Boolean(body.key),
      _key: body.key,
    };
    providers.set(id, provider);
    return HttpResponse.json(toResponse(provider), { status: 201 });
  }),

  // GET /api/llm/model-providers/:id
  http.get("*/api/llm/model-providers/:id", ({ params }) => {
    const provider = providers.get(String(params.id));
    if (!provider) {
      return HttpResponse.json({ detail: "Not found" }, { status: 404 });
    }
    return HttpResponse.json(toResponse(provider));
  }),

  // PATCH /api/llm/model-providers/:id
  http.patch("*/api/llm/model-providers/:id", async ({ request, params }) => {
    const provider = providers.get(String(params.id));
    if (!provider) {
      return HttpResponse.json({ detail: "Not found" }, { status: 404 });
    }
    const body = (await request.json()) as {
      display_name?: string;
      kind?: string;
      key?: string;
      base_url?: string;
      wire_api?: "auto" | "chat" | "responses";
      custom_headers?: Record<string, string>;
    } | null;
    if (body?.display_name !== undefined) {
      provider.display_name = body.display_name;
    }
    if (body?.kind !== undefined) provider.kind = body.kind;
    if (body?.key) {
      provider._key = body.key;
      provider.api_key_set = true;
    }
    if (body?.base_url !== undefined) provider.base_url = body.base_url;
    if (body?.wire_api !== undefined) provider.wire_api = body.wire_api;
    if (body?.custom_headers !== undefined) {
      provider.custom_headers = body.custom_headers;
    }
    provider.updated_at = now();
    return HttpResponse.json(toResponse(provider));
  }),

  // DELETE /api/llm/model-providers/:id
  http.delete("*/api/llm/model-providers/:id", ({ params }) => {
    const provider = providers.get(String(params.id));
    if (!provider) {
      return HttpResponse.json({ detail: "Not found" }, { status: 404 });
    }
    providers.delete(provider.id);
    return HttpResponse.json(toResponse(provider));
  }),

  // POST /api/llm/model-providers/:id/models — add a model
  http.post(
    "*/api/llm/model-providers/:id/models",
    async ({ request, params }) => {
      const provider = providers.get(String(params.id));
      if (!provider) {
        return HttpResponse.json({ detail: "Not found" }, { status: 404 });
      }
      const body = (await request.json()) as {
        name?: string;
        wire_api?: "auto" | "chat" | "responses" | null;
      } | null;
      if (!body?.name) {
        return HttpResponse.json(
          { detail: "name is required" },
          { status: 400 },
        );
      }
      if (provider.models.some((m) => m.name === body.name)) {
        return HttpResponse.json(
          { detail: "Model already exists" },
          { status: 409 },
        );
      }
      provider.models.push({
        name: body.name,
        wire_api: body.wire_api ?? null,
      });
      provider.updated_at = now();
      return HttpResponse.json(toResponse(provider), { status: 201 });
    },
  ),

  // PATCH /api/llm/model-providers/:id/models/:model — edit a model
  http.patch(
    "*/api/llm/model-providers/:id/models/:model",
    async ({ request, params }) => {
      const provider = providers.get(String(params.id));
      if (!provider) {
        return HttpResponse.json({ detail: "Not found" }, { status: 404 });
      }
      const existing = provider.models.find(
        (m) => m.name === String(params.model),
      );
      if (!existing) {
        return HttpResponse.json(
          { detail: "Model not found" },
          { status: 404 },
        );
      }
      const body = (await request.json()) as {
        name?: string;
        wire_api?: "auto" | "chat" | "responses" | null;
      } | null;
      if (body?.name) existing.name = body.name;
      if (body?.wire_api !== undefined) existing.wire_api = body.wire_api;
      provider.updated_at = now();
      return HttpResponse.json(toResponse(provider));
    },
  ),

  // DELETE /api/llm/model-providers/:id/models/:model — remove a model
  http.delete("*/api/llm/model-providers/:id/models/:model", ({ params }) => {
    const provider = providers.get(String(params.id));
    if (!provider) {
      return HttpResponse.json({ detail: "Not found" }, { status: 404 });
    }
    provider.models = provider.models.filter(
      (m) => m.name !== String(params.model),
    );
    provider.updated_at = now();
    return HttpResponse.json(toResponse(provider));
  }),

  // POST /api/llm/model-providers/:id/test — probe the stored key
  http.post("*/api/llm/model-providers/:id/test", ({ params }) => {
    const provider = providers.get(String(params.id));
    if (!provider) {
      return HttpResponse.json({ detail: "Not found" }, { status: 404 });
    }
    const ok = Boolean(provider._key) || provider.api_key_set;
    const suggested =
      provider.kind === "openai"
        ? ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "o3-mini"]
        : ["mock-model-1"];
    return HttpResponse.json({
      id: provider.id,
      ok,
      verified: ok,
      suggested_models: ok ? suggested : [],
      error: ok ? null : "Invalid API key",
    });
  }),
];
