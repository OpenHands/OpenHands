import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { getCloudSuggestedTasks } from "#/api/cloud/suggestions-service.api";
import type { Backend } from "#/api/backend-registry/types";
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
  __resetActiveStoreForTests();
  setRegisteredBackends([cloudBackend]);
  setActiveSelection({ backendId: cloudBackend.id });
  fetchMock.mockReset();
  fetchMock.mockResolvedValue(
    mockJsonResponse({ items: [], next_page_id: null }),
  );
  global.fetch = fetchMock as typeof fetch;
});

afterEach(() => {
  __resetActiveStoreForTests();
  fetchMock.mockReset();
  global.fetch = originalFetch;
});

describe("getCloudSuggestedTasks", () => {
  it("forwards limit and pageId to the upstream /api/v1/git/suggested-tasks/search endpoint", async () => {
    // Act
    await getCloudSuggestedTasks({ limit: 10, pageId: "p2" });

    // Assert
    const [url, init] = getFetchCall(fetchMock);
    expect(init).toMatchObject({
      method: "GET",
      headers: { Authorization: "Bearer bearer-token" },
    });
    expect(url).toContain("/api/v1/git/suggested-tasks/search");
    expect(url).toContain("limit=10");
    expect(url).toContain("page_id=p2");
  });
});
