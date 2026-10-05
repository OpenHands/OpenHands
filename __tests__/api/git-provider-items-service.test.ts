import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { GitProviderItemsService } from "#/api/git-provider-items-service";
import { getFetchCall, mockJsonResponse } from "./cloud/fetch-test-utils";

// A cloud backend keeps provider tokens server-side, so the service never
// touches the local secrets store and the provider REST call is the only
// fetch in play.
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
  global.fetch = fetchMock as unknown as typeof fetch;
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  fetchMock.mockReset();
  global.fetch = originalFetch;
});

describe("GitProviderItemsService", () => {
  it("lists Forgejo pull requests from the Forgejo host, not api.github.com", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse([
        {
          id: 11,
          number: 7,
          title: "Fix the thing",
          html_url: "https://codeberg.org/acme/widgets/pulls/7",
          user: { login: "octocat" },
          updated_at: "2026-09-01T10:00:00Z",
        },
      ]),
    );

    const items = await GitProviderItemsService.listPullRequests(
      "acme/widgets",
      "forgejo",
    );

    const [url] = getFetchCall(fetchMock);
    expect(url.startsWith("https://codeberg.org/api/v1/")).toBe(true);
    expect(url).toContain("/repos/acme/widgets/pulls");
    expect(items).toEqual([
      {
        id: 11,
        number: 7,
        title: "Fix the thing",
        url: "https://codeberg.org/acme/widgets/pulls/7",
        authorLogin: "octocat",
        updatedAt: "2026-09-01T10:00:00Z",
      },
    ]);
  });

  it("lists Forgejo issues from the Forgejo host and skips pull requests", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse([
        {
          id: 21,
          number: 3,
          title: "Real issue",
          html_url: "https://codeberg.org/acme/widgets/issues/3",
          user: { login: "octocat" },
          updated_at: null,
        },
        {
          id: 22,
          number: 4,
          title: "A PR returned by the issues endpoint",
          html_url: "https://codeberg.org/acme/widgets/pulls/4",
          pull_request: { merged: false },
        },
      ]),
    );

    const items = await GitProviderItemsService.listIssues(
      "acme/widgets",
      "forgejo",
    );

    const [url] = getFetchCall(fetchMock);
    expect(url.startsWith("https://codeberg.org/api/v1/")).toBe(true);
    expect(url).toContain("/repos/acme/widgets/issues");
    expect(items).toHaveLength(1);
    expect(items[0].title).toBe("Real issue");
  });
});
