import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import AgentsService from "#/api/agents-service";
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

describe("AgentsService.getAgents against cloud backend", () => {
  it("paginates /api/v1/agents/search and returns the merged list", async () => {
    fetchMock
      .mockResolvedValueOnce(
        mockJsonResponse({
          items: [
            { name: "general-purpose", level: "builtin" },
            { name: "changelog-writer", level: "project" },
          ],
          next_page_id: "changelog-writer",
        }),
      )
      .mockResolvedValueOnce(
        mockJsonResponse({
          items: [{ name: "my-helper", level: "user" }],
          next_page_id: null,
        }),
      );

    const agents = await AgentsService.getAgents();

    expect(fetchMock).toHaveBeenCalledTimes(2);

    const [firstUrl, firstInit] = getFetchCall(fetchMock, 0);
    expect(firstInit).toMatchObject({
      method: "GET",
      headers: { Authorization: "Bearer bearer-token" },
    });
    expect(firstUrl).toMatch(
      /^https:\/\/app\.all-hands\.dev\/api\/v1\/agents\/search\?/,
    );
    expect(firstUrl).not.toContain("page_id=");

    // Second page request carries the cursor from the first response.
    const [secondUrl] = getFetchCall(fetchMock, 1);
    expect(secondUrl).toContain("page_id=changelog-writer");

    expect(agents.map((a) => a.name)).toEqual([
      "general-purpose",
      "changelog-writer",
      "my-helper",
    ]);
  });

  it("returns an empty list when the cloud returns no agents", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ items: [], next_page_id: null }),
    );

    const agents = await AgentsService.getAgents();

    expect(agents).toEqual([]);
  });
});
