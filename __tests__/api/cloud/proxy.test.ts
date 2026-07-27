import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { callCloudProxy } from "#/api/cloud/proxy";
import type { Backend } from "#/api/backend-registry/types";

const cloudPersonal: Backend = {
  id: "cloud-personal",
  name: "Production - Personal",
  host: "https://app.all-hands.dev",
  apiKey: "personal-key",
  kind: "cloud",
};

const cloudAcme: Backend = {
  id: "cloud-acme",
  name: "Production - Acme",
  host: "https://app.all-hands.dev",
  apiKey: "acme-key",
  kind: "cloud",
};

const cookieCloud: Backend = {
  id: "cloud-cookie",
  name: "Production - Cookie",
  host: "https://app.all-hands.dev",
  apiKey: "",
  kind: "cloud",
  authMode: "cookie",
};

const originalFetch = global.fetch;
const fetchMock = vi.fn();

function mockJsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json" },
  });
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
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

describe("callCloudProxy X-Org-Id injection", () => {
  it("sends X-Org-Id when targeting the active cloud backend with a selected orgId", async () => {
    // Arrange — active selection points at the cloud backend with a
    // resolved orgId. This is the steady-state case after the user picks
    // an org row in the BackendSelector.
    setRegisteredBackends([cloudPersonal]);
    setActiveSelection({
      backendId: cloudPersonal.id,
      orgId: "org-personal-uuid",
    });

    // Act
    await callCloudProxy({
      backend: cloudPersonal,
      method: "GET",
      path: "/api/v1/app-conversations/search",
    });

    // Assert — the request carries the X-Org-Id of the active selection so the
    // cloud backend can scope this request to the user's locally-chosen org
    // without depending on user.current_org_id.
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe(`${cloudPersonal.host}/api/v1/app-conversations/search`);
    expect(init).toMatchObject({
      method: "GET",
    });
    expect(
      (init as { headers: Record<string, string> }).headers["X-Org-Id"],
    ).toBe("org-personal-uuid");
  });

  it("omits X-Org-Id when targeting a different cloud backend than the active one", async () => {
    // Arrange — the BackendSelector fan-out (e.g. useAllCloudOrganizations)
    // calls callCloudProxy(b) for every registered cloud backend. Sending
    // the active backend's orgId across an unrelated API key would cause
    // the cloud backend to 403 on api_key_org_id / X-Org-Id mismatch.
    setRegisteredBackends([cloudPersonal, cloudAcme]);
    setActiveSelection({
      backendId: cloudPersonal.id,
      orgId: "org-personal-uuid",
    });

    // Act — request targets the non-active backend.
    await callCloudProxy({
      backend: cloudAcme,
      method: "GET",
      path: "/api/keys/current",
    });

    // Assert
    const [, init] = fetchMock.mock.calls[0]!;
    expect(
      (init as { headers: Record<string, string> }).headers,
    ).not.toHaveProperty("X-Org-Id");
  });
});

describe("callCloudProxy cookie auth", () => {
  it("omits bearer auth for same-origin cookie-backed Cloud requests", async () => {
    setRegisteredBackends([cookieCloud]);
    setActiveSelection({ backendId: cookieCloud.id, orgId: null });

    await callCloudProxy({
      backend: cookieCloud,
      method: "GET",
      path: "/api/v1/users/me",
    });

    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe(`${cookieCloud.host}/api/v1/users/me`);
    expect(init).toMatchObject({ method: "GET" });
    expect(
      (init as { headers: Record<string, string> }).headers,
    ).not.toHaveProperty("Authorization");
  });

  it("omits bearer auth in runtime proxy envelope requests", async () => {
    setRegisteredBackends([cookieCloud]);
    setActiveSelection({ backendId: cookieCloud.id, orgId: null });

    await callCloudProxy({
      backend: cookieCloud,
      method: "GET",
      path: "/api/bash/bash_events/search",
      hostOverride: "https://abc123.prod-runtime.all-hands.dev",
    });

    const [, init] = fetchMock.mock.calls[0]!;
    const envelope = JSON.parse((init as { body: string }).body);
    expect(
      (envelope as { headers: Record<string, string> }).headers,
    ).not.toHaveProperty("Authorization");
  });
});

describe("callCloudProxy automation direct routing", () => {
  it("sends automation requests straight to the cloud host with the API key instead of the /api/cloud-proxy envelope", async () => {
    // Arrange — the automation service grants permissive CORS to API-key
    // requests (automation#185), so app-host automation calls no longer
    // need the same-origin proxy hop through the bundled agent-server.
    setRegisteredBackends([cloudPersonal]);
    setActiveSelection({ backendId: cloudPersonal.id, orgId: null });
    const page = { automations: [], total: 0 };
    fetchMock.mockResolvedValueOnce(mockJsonResponse(page));

    // Act
    const result = await callCloudProxy({
      backend: cloudPersonal,
      method: "GET",
      path: "/api/automation/v1?limit=50&offset=0",
    });

    // Assert — the browser calls the automation API on the cloud host
    // directly, authenticated by the backend's API key, and no envelope
    // POST reaches /api/cloud-proxy.
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe(
      `${cloudPersonal.host}/api/automation/v1?limit=50&offset=0`,
    );
    expect(init).toMatchObject({
      method: "GET",
    });
    expect(
      (init as { headers: Record<string, string> }).headers.Authorization,
    ).toBe(`Bearer ${cloudPersonal.apiKey}`);
    expect(result).toEqual(page);
  });

  it("forwards the blob responseType and fail-fast timeout to the direct request", async () => {
    // Arrange — tarball downloads and health probes rely on these
    // per-request options surviving the switch from the proxy envelope to
    // the direct call.
    setRegisteredBackends([cloudPersonal]);
    setActiveSelection({ backendId: cloudPersonal.id, orgId: null });

    // Act
    await callCloudProxy({
      backend: cloudPersonal,
      method: "GET",
      path: "/api/automation/v1/auto-1/tarball",
      responseType: "blob",
      timeoutSeconds: 5,
    });

    // Assert
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe(`${cloudPersonal.host}/api/automation/v1/auto-1/tarball`);
    expect(init).toMatchObject({ method: "GET" });
  });
});

describe("callCloudProxy hostOverride routing", () => {
  const runtimeHost = "https://abc123.prod-runtime.all-hands.dev";

  it("routes through the local /api/cloud-proxy instead of the upstream host when hostOverride is set", async () => {
    // Arrange — runtime-sandbox endpoints need the proxy hop because the
    // per-conversation runtime hosts reject browser requests from the
    // local GUI origin.
    setRegisteredBackends([cloudPersonal]);
    setActiveSelection({ backendId: cloudPersonal.id, orgId: null });
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ items: [] }));

    // Act
    const result = await callCloudProxy({
      backend: cloudPersonal,
      method: "GET",
      path: "/api/bash/bash_events/search",
      hostOverride: runtimeHost,
    });

    // Assert — the browser only makes a same-origin POST to the bundled
    // agent-server's proxy endpoint carrying the upstream call as an
    // envelope, and the upstream payload is unwrapped for the caller.
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toMatch(/\/api\/cloud-proxy$/);
    const envelope = JSON.parse((init as { body: string }).body);
    expect(envelope).toMatchObject({
      host: runtimeHost,
      method: "GET",
      path: "/api/bash/bash_events/search",
    });
    expect(result).toEqual({ items: [] });
  });

  it("carries bearer auth and X-Org-Id inside the proxy envelope", async () => {
    // Arrange — org scoping must survive the server-side hop: the envelope
    // headers are what the agent-server attaches to the upstream call in
    // place of the headers a direct browser request would have sent.
    setRegisteredBackends([cloudPersonal]);
    setActiveSelection({
      backendId: cloudPersonal.id,
      orgId: "org-personal-uuid",
    });

    // Act
    await callCloudProxy({
      backend: cloudPersonal,
      method: "GET",
      path: "/api/bash/bash_events/search",
      hostOverride: runtimeHost,
    });

    // Assert
    const [, init] = fetchMock.mock.calls[0]!;
    const envelope = JSON.parse((init as { body: string }).body);
    expect(
      (envelope as { headers: Record<string, string> }).headers,
    ).toMatchObject({
      Authorization: `Bearer ${cloudPersonal.apiKey}`,
      "X-Org-Id": "org-personal-uuid",
    });
  });
});
