import { http, HttpResponse } from "msw";
import { CustomSecret, GetSecretsResponse } from "#/api/secrets-service.types";

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
  http.get("/api/secrets", async () => {
    const secretsArray = Array.from(secrets.values());
    const secretsWithoutValue: Omit<CustomSecret, "value">[] = secretsArray.map(
      ({ value, ...rest }) => rest,
    );

    const data: GetSecretsResponse = {
      custom_secrets: secretsWithoutValue,
    };

    return HttpResponse.json(data);
  }),

  http.post("/api/secrets", async ({ request }) => {
    const body = (await request.json()) as CustomSecret;
    if (typeof body === "object" && body?.name) {
      secrets.set(body.name, body);
      return HttpResponse.json(true);
    }

    return HttpResponse.json(false, { status: 400 });
  }),

  http.put("/api/secrets/:id", async ({ params, request }) => {
    const { id } = params;
    const body = (await request.json()) as Omit<CustomSecret, "value">;

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

  http.delete("/api/secrets/:id", async ({ params }) => {
    const { id } = params;

    if (typeof id === "string") {
      secrets.delete(id);
      return HttpResponse.json(true);
    }

    return HttpResponse.json(false, { status: 400 });
  }),
];
