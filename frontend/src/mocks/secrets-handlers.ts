import { http, HttpResponse } from "msw";
import {
  CustomSecret,
  CustomSecretPage,
  CustomSecretWithoutValue,
  GetSecretsResponse,
} from "#/api/secrets-service.types";

const DEFAULT_SECRETS: CustomSecret[] = [
  {
    name: "OpenAI_API_Key",
    value: "test-123",
    description: "OpenAI API Key",
  },
  {
    name: "Google_Maps_API_Key",
    value: "test-123",
    description: "Google Maps API Key",
  },
];

const secrets = new Map<string, CustomSecret>(
  DEFAULT_SECRETS.map((secret) => [secret.name, secret]),
);

export const SECRETS_HANDLERS = [
  // V1 API - Search endpoint with pagination
  http.get("/api/v1/secrets/search", async ({ request }) => {
    const url = new URL(request.url);
    const nameContains = url.searchParams.get("name__contains");
    const pageId = url.searchParams.get("page_id");
    const limit = parseInt(url.searchParams.get("limit") || "100", 10);

    // Get all secrets and filter by name if needed
    let secretsArray = Array.from(secrets.values());
    if (nameContains) {
      secretsArray = secretsArray.filter((s) =>
        s.name.toLowerCase().includes(nameContains.toLowerCase()),
      );
    }

    // Sort alphabetically for consistent pagination
    secretsArray.sort((a, b) => a.name.localeCompare(b.name));

    // Apply pagination
    let startIndex = 0;
    if (pageId) {
      const pageIndex = secretsArray.findIndex((s) => s.name === pageId);
      if (pageIndex >= 0) {
        startIndex = pageIndex + 1;
      }
    }

    const pageItems = secretsArray.slice(startIndex, startIndex + limit);
    const hasMore = startIndex + limit < secretsArray.length;

    const items: CustomSecretWithoutValue[] = pageItems.map(
      ({ value, ...rest }) => rest,
    );

    const data: CustomSecretPage = {
      items,
      next_page_id: hasMore ? (items[items.length - 1]?.name ?? null) : null,
    };

    return HttpResponse.json(data);
  }),

  // Legacy V0 API - deprecated but kept for compatibility
  http.get("/api/secrets", async () => {
    const secretsArray = Array.from(secrets.values());
    const secretsWithoutValue: CustomSecretWithoutValue[] = secretsArray.map(
      ({ value, ...rest }) => rest,
    );

    const data: GetSecretsResponse = {
      custom_secrets: secretsWithoutValue,
    };

    return HttpResponse.json(data);
  }),

  // V1 API - Create secret
  http.post("/api/v1/secrets", async ({ request }) => {
    const body = (await request.json()) as CustomSecret;
    if (typeof body === "object" && body?.name) {
      secrets.set(body.name, body);
      return new HttpResponse(null, { status: 201 });
    }

    return HttpResponse.json(false, { status: 400 });
  }),

  // Legacy V0 API - Create secret (deprecated)
  http.post("/api/secrets", async ({ request }) => {
    const body = (await request.json()) as CustomSecret;
    if (typeof body === "object" && body?.name) {
      secrets.set(body.name, body);
      return new HttpResponse(null, { status: 201 });
    }

    return HttpResponse.json(false, { status: 400 });
  }),

  // V1 API - Update secret
  http.put("/api/v1/secrets/:id", async ({ params, request }) => {
    const { id } = params;
    const body = (await request.json()) as CustomSecretWithoutValue;

    if (typeof id === "string" && typeof body === "object") {
      const secret = secrets.get(id);
      if (secret && body?.name) {
        const newSecret: CustomSecret = { ...secret, ...body };
        secrets.delete(id);
        secrets.set(body.name, newSecret);
        return HttpResponse.json(true);
      }
    }

    return HttpResponse.json(false, { status: 400 });
  }),

  // Legacy V0 API - Update secret (deprecated)
  http.put("/api/secrets/:id", async ({ params, request }) => {
    const { id } = params;
    const body = (await request.json()) as CustomSecretWithoutValue;

    if (typeof id === "string" && typeof body === "object") {
      const secret = secrets.get(id);
      if (secret && body?.name) {
        const newSecret: CustomSecret = { ...secret, ...body };
        secrets.delete(id);
        secrets.set(body.name, newSecret);
        return HttpResponse.json(true);
      }
    }

    return HttpResponse.json(false, { status: 400 });
  }),

  // V1 API - Delete secret
  http.delete("/api/v1/secrets/:id", async ({ params }) => {
    const { id } = params;

    if (typeof id === "string") {
      secrets.delete(id);
      return HttpResponse.json(true);
    }

    return HttpResponse.json(false, { status: 400 });
  }),

  // Legacy V0 API - Delete secret (deprecated)
  http.delete("/api/secrets/:id", async ({ params }) => {
    const { id } = params;

    if (typeof id === "string") {
      secrets.delete(id);
      return HttpResponse.json(true);
    }

    return HttpResponse.json(false, { status: 400 });
  }),
];
