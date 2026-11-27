import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import ProfilesService, {
  type SaveProfileRequest,
} from "#/api/profiles-service/profiles-service.api";
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
const ORG_BASE = `https://app.all-hands.dev/api/organizations/${ORG_ID}/profiles`;
const SETTINGS_BASE = "https://app.all-hands.dev/api/v1/settings/profiles";
const originalFetch = global.fetch;
const fetchMock = vi.fn();

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

// With an org bound, profile CRUD goes through the org-gated routes so the
// server enforces EDIT_ORG_SETTINGS (a member's mutation 403s, not just hidden).
describe("ProfilesService against a cloud org (gated org routes)", () => {
  beforeEach(() => {
    setActiveSelection({ backendId: cloudBackend.id, orgId: ORG_ID });
  });

  it("lists profiles via GET /api/organizations/{orgId}/profiles", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        profiles: [
          { name: "gpt", model: "gpt-4o", base_url: null, api_key_set: true },
        ],
        active_profile: "gpt",
      }),
    );

    const res = await ProfilesService.listProfiles();

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(ORG_BASE);
    expect(init).toMatchObject({
      method: "GET",
      headers: { Authorization: "Bearer bearer-token" },
    });
    expect(res.active_profile).toBe("gpt");
  });

  it("fetches a profile and maps the org `llm` onto `config`", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        name: "my profile",
        llm: { model: "gpt-4o", api_key: null },
      }),
    );

    const res = await ProfilesService.getProfile("my profile");

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(`${ORG_BASE}/my%20profile`);
    expect(init).toMatchObject({
      method: "GET",
    });
    expect(res).toEqual({
      name: "my profile",
      config: { model: "gpt-4o", api_key: null },
      api_key_set: false,
    });
  });

  it("saves a profile via POST .../{name} forwarding the request body", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ name: "gpt", message: "Profile 'gpt' saved" }),
    );

    await ProfilesService.saveProfile("gpt", {
      llm: { model: "gpt-4o" } as SaveProfileRequest["llm"],
      include_secrets: true,
    });

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(`${ORG_BASE}/gpt`);
    expect(init).toMatchObject({
      method: "POST",
    });
    expect(getJsonBody(init)).toEqual({
      llm: { model: "gpt-4o" },
      include_secrets: true,
    });
  });

  it("deletes a profile via DELETE .../{name}", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ name: "gpt", message: "Profile 'gpt' deleted" }),
    );

    await ProfilesService.deleteProfile("gpt");

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(`${ORG_BASE}/gpt`);
    expect(init).toMatchObject({ method: "DELETE" });
  });

  it("renames a profile via POST .../{name}/rename with new_name", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ name: "new", message: "renamed" }),
    );

    await ProfilesService.renameProfile("old", "new");

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(`${ORG_BASE}/old/rename`);
    expect(init).toMatchObject({
      method: "POST",
    });
    expect(getJsonBody(init)).toEqual({ new_name: "new" });
  });

  it("activates a profile and maps the org `llm` onto `llm_applied`", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        name: "gpt",
        message: "Switched to profile 'gpt'",
        llm: { model: "gpt-4o" },
      }),
    );

    const res = await ProfilesService.activateProfile("gpt");

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(`${ORG_BASE}/gpt/activate`);
    expect(init).toMatchObject({
      method: "POST",
    });
    expect(res).toEqual({
      name: "gpt",
      message: "Switched to profile 'gpt'",
      llm_applied: true,
    });
  });
});

// Legacy API keys have no org bound; CRUD falls back to the per-user settings
// route (ungated — there is no org role to enforce against).
describe("ProfilesService on a cloud backend with no org (fallback)", () => {
  beforeEach(() => {
    setActiveSelection({ backendId: cloudBackend.id });
  });

  it("lists via the per-user settings route", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ profiles: [], active_profile: null }),
    );

    await ProfilesService.listProfiles();

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(SETTINGS_BASE);
    expect(init).toMatchObject({ method: "GET" });
  });

  it("saves via the per-user settings route", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ name: "gpt", message: "saved" }),
    );

    await ProfilesService.saveProfile("gpt", {
      llm: { model: "gpt-4o" } as SaveProfileRequest["llm"],
      include_secrets: true,
    });

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(`${SETTINGS_BASE}/gpt`);
    expect(init).toMatchObject({ method: "POST" });
  });

  it("activates via the per-user settings route and maps `model`", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ name: "gpt", message: "ok", model: "gpt-4o" }),
    );

    const res = await ProfilesService.activateProfile("gpt");

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(`${SETTINGS_BASE}/gpt/activate`);
    expect(init).toMatchObject({
      method: "POST",
    });
    expect(res.llm_applied).toBe(true);
  });
});
