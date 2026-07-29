import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { getCloudOrganizationMe } from "#/api/cloud/organization-service.api";
import { getFetchCall, mockJsonResponse } from "./fetch-test-utils";

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
  global.fetch = fetchMock as typeof fetch;
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  fetchMock.mockReset();
  global.fetch = originalFetch;
});

describe("cloud organization /me", () => {
  it("calls /api/organizations/{orgId}/me directly and returns user_id", async () => {
    const orgId = "0b93b5f2-5396-49f2-8d98-61f906184270";
    fetchMock.mockResolvedValue(
      mockJsonResponse({
        org_id: orgId,
        user_id: orgId,
        email: "hieptl.developer@gmail.com",
        role: "owner",
        permissions: ["view_org_settings", "edit_org_settings"],
      }),
    );

    const result = await getCloudOrganizationMe(orgId);

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(`${cloudBackend.host}/api/organizations/${orgId}/me`);
    expect(init).toMatchObject({
      method: "GET",
      headers: { Authorization: "Bearer bearer-token" },
    });
    expect(result).toEqual({
      orgId,
      userId: orgId,
      role: "owner",
      permissions: ["view_org_settings", "edit_org_settings"],
    });
  });

  it("returns null permissions when the app-server omits them (older version)", async () => {
    const orgId = "0b93b5f2-5396-49f2-8d98-61f906184270";
    fetchMock.mockResolvedValue(
      mockJsonResponse({
        org_id: orgId,
        user_id: orgId,
        email: "x@example.com",
        role: "member",
      }),
    );

    const result = await getCloudOrganizationMe(orgId);

    // Absent `permissions` → null, so callers fall back to the role check.
    expect(result).toEqual({
      orgId,
      userId: orgId,
      role: "member",
      permissions: null,
    });
  });
});
