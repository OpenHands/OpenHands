import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { server } from "#/mocks/node";

const SECRETS_URL = "http://localhost:3000/api/settings/secrets";
const UNEXPECTED_REQUEST_HANDLER = http.all(
  "*",
  () => new HttpResponse(null, { status: 599 }),
);

const secretUrl = (name: string) =>
  `${SECRETS_URL}/${encodeURIComponent(name)}`;

beforeEach(async () => {
  vi.resetModules();
  const { SECRETS_HANDLERS } = await import("#/mocks/secrets-handlers");
  server.resetHandlers(...SECRETS_HANDLERS, UNEXPECTED_REQUEST_HANDLER);
});

describe("mock secrets API", () => {
  it("lists secret metadata without exposing stored values", async () => {
    const response = await fetch(SECRETS_URL);
    const body = (await response.json()) as {
      secrets: Array<Record<string, unknown>>;
    };

    expect(response.status).toBe(200);
    expect(body).toEqual({
      secrets: [
        { name: "OpenAI_API_Key", description: "OpenAI API Key" },
        {
          name: "Google_Maps_API_Key",
          description: "Google Maps API Key",
        },
      ],
    });
    expect(body.secrets.every((secret) => !("value" in secret))).toBe(true);
  });

  it("returns a stored value as plain text and reports an unknown name", async () => {
    const found = await fetch(secretUrl("OpenAI_API_Key"));

    expect(found.status).toBe(200);
    expect(found.headers.get("content-type")).toBe("text/plain");
    await expect(found.text()).resolves.toBe("test-123");

    const secondSeed = await fetch(secretUrl("Google_Maps_API_Key"));

    expect(secondSeed.status).toBe(200);
    await expect(secondSeed.text()).resolves.toBe("test-123");

    const missing = await fetch(secretUrl("unknown-secret"));

    expect(missing.status).toBe(404);
    await expect(missing.json()).resolves.toEqual({
      detail: "Secret not found",
    });
  });

  it("persists create, update, list, read, and delete transitions", async () => {
    const name = "Coverage_API_Key";

    try {
      const created = await fetch(SECRETS_URL, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          value: "initial-value",
          description: "Coverage secret",
        }),
      });

      expect(created.status).toBe(200);
      await expect(created.json()).resolves.toEqual({
        name,
        description: "Coverage secret",
      });

      const listed = await fetch(SECRETS_URL);
      const listedBody = (await listed.json()) as {
        secrets: Array<Record<string, unknown>>;
      };
      expect(listedBody.secrets).toContainEqual({
        name,
        description: "Coverage secret",
      });
      expect(
        listedBody.secrets.find((secret) => secret.name === name),
      ).not.toHaveProperty("value");

      const firstRead = await fetch(secretUrl(name));
      await expect(firstRead.text()).resolves.toBe("initial-value");

      const updated = await fetch(SECRETS_URL, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, value: "updated-value" }),
      });
      await expect(updated.json()).resolves.toEqual({ name });

      const secondRead = await fetch(secretUrl(name));
      await expect(secondRead.text()).resolves.toBe("updated-value");

      const deleted = await fetch(secretUrl(name), { method: "DELETE" });
      expect(deleted.status).toBe(200);
      await expect(deleted.json()).resolves.toEqual({ deleted: true });

      const afterDelete = await fetch(secretUrl(name));
      expect(afterDelete.status).toBe(404);

      const deleteAgain = await fetch(secretUrl(name), { method: "DELETE" });
      expect(deleteAgain.status).toBe(404);
      await expect(deleteAgain.json()).resolves.toEqual({
        detail: "Secret not found",
      });
    } finally {
      await fetch(secretUrl(name), { method: "DELETE" });
    }
  });

  it.each([
    ["null body", null],
    ["missing name", { value: "secret-value" }],
    ["missing value", { name: "Incomplete_Secret" }],
  ])("rejects an upsert with a %s", async (_case, payload) => {
    const response = await fetch(SECRETS_URL, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({
      detail: "name and value are required",
    });
  });
});
