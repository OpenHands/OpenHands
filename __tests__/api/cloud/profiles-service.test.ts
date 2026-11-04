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
const OPTIONS_BASE = "https://app.all-hands.dev/api/v1/agent-profile/options";
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

// With an org bound, the list view uses the combined Enterprise options
// endpoint, while CRUD writes personal LLM profiles through settings routes.
describe("ProfilesService against a cloud org", () => {
  beforeEach(() => {
    setActiveSelection({ backendId: cloudBackend.id, orgId: ORG_ID });
  });

  it("lists profiles via the combined agent profile options endpoint", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        llm_profiles: [
          { name: "gpt", model: "gpt-4o", base_url: null, api_key_set: true },
        ],
      }),
    );

    const res = await ProfilesService.listProfiles();

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(OPTIONS_BASE);
    expect(init).toMatchObject({
      method: "GET",
      headers: { Authorization: "Bearer bearer-token" },
    });
    expect(res).toEqual({
      profiles: [
        {
          name: "gpt",
          model: "gpt-4o",
          base_url: null,
          api_key_set: true,
          provider_connection_id: null,
        },
      ],
      active_profile: null,
    });
  });

  it("fetches a personal profile through settings routes", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        name: "my profile",
        config: { model: "gpt-4o", api_key: null },
      }),
    );

    const res = await ProfilesService.getProfile("my profile");

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(`${SETTINGS_BASE}/my%20profile`);
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
    expect(url).toBe(`${SETTINGS_BASE}/gpt`);
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
    expect(url).toBe(`${SETTINGS_BASE}/gpt`);
    expect(init).toMatchObject({ method: "DELETE" });
  });

  it("renames a profile via POST .../{name}/rename with new_name", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ name: "new", message: "renamed" }),
    );

    await ProfilesService.renameProfile("old", "new");

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(`${SETTINGS_BASE}/old/rename`);
    expect(init).toMatchObject({
      method: "POST",
    });
    expect(getJsonBody(init)).toEqual({ new_name: "new" });
  });

  it("activates a personal profile and maps `model` onto `llm_applied`", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        name: "gpt",
        message: "Switched to profile 'gpt'",
        model: "gpt-4o",
      }),
    );

    const res = await ProfilesService.activateProfile("gpt");

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(`${SETTINGS_BASE}/gpt/activate`);
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

// Legacy API keys have no org bound; options may be unavailable, so listing
// falls back to the per-user settings route.
describe("ProfilesService on a cloud backend with no org (fallback)", () => {
  beforeEach(() => {
    setActiveSelection({ backendId: cloudBackend.id });
  });

  it("falls back to the per-user settings route when options are unavailable", async () => {
    fetchMock
      .mockRejectedValueOnce(new Error("not found"))
      .mockResolvedValueOnce(
        mockJsonResponse({ profiles: [], active_profile: null }),
      );

    await ProfilesService.listProfiles();

    const [optionsUrl, optionsInit] = getFetchCall(fetchMock, 0);
    expect(optionsUrl).toBe(OPTIONS_BASE);
    expect(optionsInit).toMatchObject({ method: "GET" });

    const [settingsUrl, settingsInit] = getFetchCall(fetchMock, 1);
    expect(settingsUrl).toBe(SETTINGS_BASE);
    expect(settingsInit).toMatchObject({ method: "GET" });
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
