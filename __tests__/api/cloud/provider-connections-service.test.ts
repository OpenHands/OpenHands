import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import ProviderConnectionsService from "#/api/provider-connections-service/provider-connections-service.api";
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

const ORG_ID = "org-1";
const BASE = `https://app.all-hands.dev/api/organizations/${ORG_ID}/provider-connections`;
const originalFetch = global.fetch;
const fetchMock = vi.fn();

const connection = {
  id: "conn-1",
  display_name: "My OpenAI",
  provider: "openai",
  base_url: null,
  created_at: 1,
  updated_at: 2,
  api_key_set: true,
};

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([cloudBackend]);
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

// With an org bound, provider-connection CRUD goes through the org-gated routes
// so the server enforces EDIT_ORG_SETTINGS / VIEW_ORG_SETTINGS.
describe("ProviderConnectionsService against a cloud org", () => {
  beforeEach(() => {
    setActiveSelection({ backendId: cloudBackend.id, orgId: ORG_ID });
  });

  it("lists connections and unwraps the { connections } envelope", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ connections: [connection] }),
    );

    const res = await ProviderConnectionsService.list();

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(BASE);
    expect(init).toMatchObject({
      method: "GET",
      headers: { Authorization: "Bearer bearer-token" },
    });
    expect(res).toEqual([connection]);
  });

  it("returns [] when the org has no connections", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ connections: [] }));

    const res = await ProviderConnectionsService.list();

    expect(res).toEqual([]);
  });

  it("creates a connection via POST with the request body", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse(connection));

    const request = {
      display_name: "My OpenAI",
      provider: "openai",
      api_key: "sk-123",
      base_url: null,
    };
    const res = await ProviderConnectionsService.create(request);

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(BASE);
    expect(init).toMatchObject({ method: "POST" });
    expect(getJsonBody(init)).toEqual(request);
    expect(res).toEqual(connection);
  });

  it("updates a connection via PATCH at the id-scoped path", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse(connection));

    const res = await ProviderConnectionsService.update("conn-1", {
      display_name: "Renamed",
    });

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(`${BASE}/conn-1`);
    expect(init).toMatchObject({ method: "PATCH" });
    expect(getJsonBody(init)).toEqual({ display_name: "Renamed" });
    expect(res).toEqual(connection);
  });

  it("url-encodes the id when deleting", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse(connection));

    await ProviderConnectionsService.delete("a b/c");

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(`${BASE}/a%20b%2Fc`);
    expect(init).toMatchObject({ method: "DELETE" });
  });
});

// Without an org bound (legacy API keys), the org-scoped route is unaddressable.
describe("ProviderConnectionsService on a cloud backend with no org", () => {
  beforeEach(() => {
    setActiveSelection({ backendId: cloudBackend.id });
  });

  it("throws rather than firing an unaddressable request", async () => {
    await expect(ProviderConnectionsService.list()).rejects.toThrow(
      /organization-bound/i,
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
