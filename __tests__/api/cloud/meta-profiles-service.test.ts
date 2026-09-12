import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import MetaProfilesService, {
  type MetaProfile,
} from "#/api/meta-profiles-service/meta-profiles-service.api";
import {
  getFetchCall,
  getJsonBody,
  mockJsonResponse,
} from "./fetch-test-utils";

const backend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};
const orgId = "org-1";
const base = `${backend.host}/api/organizations/${orgId}/meta-profiles`;
const config: MetaProfile = {
  classifier_model: "classifier",
  default_model: "default",
  prompt_template: "Route {{ instance_text }}",
  model_table: null,
};
const originalFetch = global.fetch;
const fetchMock = vi.fn();

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([backend]);
  setActiveSelection({ backendId: backend.id, orgId });
  fetchMock.mockReset();
  global.fetch = fetchMock as typeof fetch;
});

afterEach(() => {
  global.fetch = originalFetch;
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("MetaProfilesService against a cloud org", () => {
  it("routes list and detail reads through the org API", async () => {
    fetchMock
      .mockResolvedValueOnce(
        mockJsonResponse({ meta_profiles: [], active_meta_profile: null }),
      )
      .mockResolvedValueOnce(mockJsonResponse({ name: "pareto", config }));

    await MetaProfilesService.listMetaProfiles();
    expect(getFetchCall(fetchMock)[0]).toBe(base);
    await MetaProfilesService.getMetaProfile("pareto");
    expect(getFetchCall(fetchMock, 1)[0]).toBe(`${base}/pareto`);
  });

  it("routes save, activation, and delete through the org API", async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(mockJsonResponse({ name: "pareto", message: "ok" })),
    );

    await MetaProfilesService.saveMetaProfile("pareto", config);
    let [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(`${base}/pareto`);
    expect(init.method).toBe("POST");
    expect(getJsonBody(init)).toEqual(config);

    await MetaProfilesService.activateMetaProfile("pareto");
    [url, init] = getFetchCall(fetchMock, 1);
    expect(url).toBe(`${base}/pareto/activate`);
    expect(init.method).toBe("POST");

    await MetaProfilesService.deleteMetaProfile("pareto");
    [url, init] = getFetchCall(fetchMock, 2);
    expect(url).toBe(`${base}/pareto`);
    expect(init.method).toBe("DELETE");
  });
});

it("rejects a cloud backend without an organization", async () => {
  setActiveSelection({ backendId: backend.id });
  await expect(MetaProfilesService.listMetaProfiles()).rejects.toThrow(
    /organization-bound/i,
  );
  expect(fetchMock).not.toHaveBeenCalled();
});
