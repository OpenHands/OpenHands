import { http, HttpResponse } from "msw";

/**
 * In-memory secrets storage for mock agent-server API.
 * Uses name as the key (agent-server uses name-based lookups, not IDs).
 */
const secrets = new Map<string, { value: string; description?: string }>([
  ["OpenAI_API_Key", { value: "test-123", description: "OpenAI API Key" }],
  [
    "Google_Maps_API_Key",
    { value: "test-123", description: "Google Maps API Key" },
  ],
]);

/**
 * Mock handlers for the agent-server secrets API.
 * Routes: /api/settings/secrets and /api/settings/secrets/:name
 *
 * Uses wildcard "*" prefix to match both relative paths and absolute URLs
 * (e.g., http://127.0.0.1:8000/api/...) since the code uses absolute URLs
 * when VITE_BACKEND_BASE_URL is configured.
 */
export const SECRETS_HANDLERS = [
  // GET /api/settings/secrets - List all secrets (names and descriptions only)
  http.get("*/api/settings/secrets", async ({ request }) => {
    // Exclude requests with a :name param (handled by the next handler)
    const url = new URL(request.url);
    const pathParts = url.pathname.split("/").filter(Boolean);
    if (pathParts.length > 3) return undefined; // Let it pass through to :name handler

    const secretsList = Array.from(secrets.entries()).map(
      ([name, { description }]) => ({ name, description }),
    );
    return HttpResponse.json({ secrets: secretsList });
  }),

  // GET /api/settings/secrets/:name - Get secret value by name
  http.get("*/api/settings/secrets/:name", async ({ params }) => {
    const { name } = params;
    if (typeof name !== "string") {
      return HttpResponse.json({ detail: "Invalid name" }, { status: 400 });
    }

    const secret = secrets.get(name);
    if (!secret) {
      return HttpResponse.json({ detail: "Secret not found" }, { status: 404 });
    }

    return new HttpResponse(secret.value, {
      headers: { "Content-Type": "text/plain" },
    });
  }),

  // PUT /api/settings/secrets - Create or update a secret (upsert)
  http.put("*/api/settings/secrets", async ({ request }) => {
    const body = (await request.json()) as {
      name: string;
      value: string;
      description?: string;
    } | null;

    if (!body?.name || !body?.value) {
      return HttpResponse.json(
        { detail: "name and value are required" },
        { status: 400 },
      );
    }

    secrets.set(body.name, {
      value: body.value,
      description: body.description,
    });

    return HttpResponse.json({
      name: body.name,
      description: body.description,
    });
  }),

  // DELETE /api/settings/secrets/:name - Delete a secret by name
  http.delete("*/api/settings/secrets/:name", async ({ params }) => {
    const { name } = params;
    if (typeof name !== "string") {
      return HttpResponse.json({ detail: "Invalid name" }, { status: 400 });
    }

    const deleted = secrets.delete(name);
    if (!deleted) {
      return HttpResponse.json({ detail: "Secret not found" }, { status: 404 });
    }

    return HttpResponse.json({ deleted: true });
  }),
];
