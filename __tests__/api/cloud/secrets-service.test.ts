import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { SecretsService } from "#/api/secrets-service";
import {
  getFetchCall,
  getJsonBody,
  mockJsonResponse,
} from "./fetch-test-utils";

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

const originalFetch = global.fetch;
const fetchMock = vi.fn();

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([cloudBackend]);
  setActiveSelection({ backendId: cloudBackend.id });
  fetchMock.mockReset();
  fetchMock.mockResolvedValue(mockJsonResponse({}));
  global.fetch = fetchMock as typeof fetch;
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  fetchMock.mockReset();
  global.fetch = originalFetch;
});

describe("SecretsService against cloud backend", () => {
  it("paginates getSecrets directly and returns the merged list", async () => {
    fetchMock
      .mockResolvedValueOnce(
        mockJsonResponse({
          items: [
            { name: "ALPHA", description: "first" },
            { name: "BETA", description: "second" },
          ],
          next_page_id: "BETA",
        }),
      )
      .mockResolvedValueOnce(
        mockJsonResponse({
          items: [{ name: "GAMMA", description: "third" }],
          next_page_id: null,
        }),
      );

    const secrets = await SecretsService.getSecrets();

    expect(fetchMock).toHaveBeenCalledTimes(2);

    const [firstUrl, firstInit] = getFetchCall(fetchMock, 0);
    expect(firstInit).toMatchObject({
      method: "GET",
      headers: { Authorization: "Bearer bearer-token" },
    });
    expect(firstUrl).toMatch(
      /^https:\/\/app\.all-hands\.dev\/api\/v1\/secrets\/search\?/,
    );
    expect(firstUrl).not.toContain("page_id=");

    const [secondUrl] = getFetchCall(fetchMock, 1);
    expect(secondUrl).toContain("page_id=BETA");

    expect(secrets.map((s) => s.name)).toEqual(["ALPHA", "BETA", "GAMMA"]);
  });

  it("creates a secret via direct POST /api/v1/secrets", async () => {
    await SecretsService.createSecret(
      "OPENAI_API_KEY",
      "sk-test",
      "OpenAI key",
    );

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(`${cloudBackend.host}/api/v1/secrets`);
    expect(init).toMatchObject({
      method: "POST",
      headers: { Authorization: "Bearer bearer-token" },
    });
    expect(getJsonBody(init)).toEqual({
      name: "OPENAI_API_KEY",
      value: "sk-test",
      description: "OpenAI key",
    });
  });

  it("updates a secret via PUT /api/v1/secrets/{id} with name + description only", async () => {
    // The form/hook calls updateSecret(secretToEdit, newName, description).
    await SecretsService.updateSecret("OLD_NAME", "NEW_NAME", "renamed");

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(`${cloudBackend.host}/api/v1/secrets/OLD_NAME`);
    expect(init).toMatchObject({
      method: "PUT",
      headers: { Authorization: "Bearer bearer-token" },
    });
    expect(getJsonBody(init)).toEqual({
      name: "NEW_NAME",
      description: "renamed",
    });
  });

  it("deletes a secret via direct DELETE /api/v1/secrets/{id}", async () => {
    await SecretsService.deleteSecret("token with space");

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(
      `${cloudBackend.host}/api/v1/secrets/token%20with%20space`,
    );
    expect(init).toMatchObject({
      method: "DELETE",
      headers: { Authorization: "Bearer bearer-token" },
    });
  });

  it("treats a delete 404 as success (secret already gone)", async () => {
    // Fresh Response per attempt: the retry helper re-fetches and a
    // Response body can only be consumed once. Fake timers skip the
    // retry backoff sleeps.
    fetchMock.mockImplementation(() =>
      Promise.resolve(mockJsonResponse({ detail: "Secret not found" }, 404)),
    );
    vi.useFakeTimers();

    try {
      const assertion = expect(
        SecretsService.deleteSecret("ALREADY_GONE"),
      ).resolves.toBeUndefined();
      await vi.runAllTimersAsync();
      await assertion;
    } finally {
      vi.useRealTimers();
    }
  });
});
