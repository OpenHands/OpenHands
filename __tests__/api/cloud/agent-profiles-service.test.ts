import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  getCloudAgentProfile,
  listCloudAgentProfiles,
} from "#/api/cloud/agent-profiles-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { getFetchCall, mockJsonResponse } from "./fetch-test-utils";

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://localhost:8000",
  apiKey: "local-key",
  kind: "local",
};

const fetchMock = vi.fn();
const originalFetch = global.fetch;

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([localBackend, cloudBackend]);
  setActiveSelection({ backendId: localBackend.id });
  fetchMock.mockReset();
  fetchMock.mockResolvedValue(mockJsonResponse({ profiles: [] }));
  global.fetch = fetchMock as typeof fetch;
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  global.fetch = originalFetch;
});

describe("cloud agent profiles for an explicit backend", () => {
  it("lists profiles on the picked cloud backend while local is active", async () => {
    await listCloudAgentProfiles(cloudBackend);

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe("https://app.all-hands.dev/api/agent-profiles");
    expect(init).toMatchObject({
      method: "GET",
      headers: { Authorization: "Bearer bearer-token" },
    });
  });

  it("reads one profile from the picked cloud backend", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ name: "fast", profile: { id: "profile-fast" } }),
    );

    await getCloudAgentProfile("fast", cloudBackend);

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe("https://app.all-hands.dev/api/agent-profiles/fast");
    expect(init).toMatchObject({ method: "GET" });
  });
});
