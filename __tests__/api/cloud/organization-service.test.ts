import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HttpError } from "@openhands/typescript-client";
import {
  __resetActiveStoreForTests,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import {
  getCloudOrganizations,
  getCurrentCloudApiKey,
} from "#/api/cloud/organization-service.api";
import type { Backend } from "#/api/backend-registry/types";

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

const originalFetch = global.fetch;
const fetchMock = vi.fn();

function mockJsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    statusText: status === 200 ? "OK" : "Error",
    headers: { "content-type": "application/json" },
  });
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([]);
  fetchMock.mockReset();
  global.fetch = fetchMock as typeof fetch;
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  fetchMock.mockReset();
  global.fetch = originalFetch;
});

describe("cloud organization-service", () => {
  it("getCloudOrganizations calls the cloud API directly and returns normalized data", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        items: [{ id: "org-1", name: "Personal" }],
        current_org_id: "org-1",
      }),
    );

    const result = await getCloudOrganizations(cloudBackend);

    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = fetchMock.mock.calls[0]!;

    expect(url).toBe(`${cloudBackend.host}/api/organizations`);
    expect(init).toMatchObject({
      method: "GET",
      headers: { Authorization: "Bearer bearer-token" },
    });

    expect(result).toEqual({
      items: [{ id: "org-1", name: "Personal" }],
      currentOrgId: "org-1",
    });
  });

  it("getCurrentCloudApiKey hits /api/keys/current and returns the bound orgId", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        id: "key-1",
        name: "k",
        org_id: "org-bound",
        user_id: "user-1",
        auth_type: "bearer",
      }),
    );

    const result = await getCurrentCloudApiKey(cloudBackend);

    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe(`${cloudBackend.host}/api/keys/current`);
    expect(init).toMatchObject({
      method: "GET",
      headers: { Authorization: "Bearer bearer-token" },
    });
    expect(result).toEqual({ orgId: "org-bound", isLegacyKey: false });
  });

  it("getCurrentCloudApiKey treats an upstream 400 as a legacy key (no binding)", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ detail: "bad" }, 400));

    const result = await getCurrentCloudApiKey(cloudBackend);

    expect(result).toEqual({ orgId: null, isLegacyKey: true });
  });

  it("getCurrentCloudApiKey rethrows non-400 upstream errors (e.g. revoked key)", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ detail: "unauthorized" }, 401),
    );

    await expect(getCurrentCloudApiKey(cloudBackend)).rejects.toBeInstanceOf(
      HttpError,
    );
  });
});
