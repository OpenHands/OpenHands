import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { SettingsClient } from "@openhands/typescript-client/clients";
import {
  GitProviderItemsService,
  normalizeProviderBaseUrl,
} from "#/api/git-provider-items-service";
import { SecretsService } from "#/api/secrets-service";
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

// A local backend reads provider tokens and self-hosted hosts from the
// agent-server secrets store.
const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://127.0.0.1:8000",
  apiKey: "session-key",
  kind: "local",
};

function useLocalSecrets(secrets: Record<string, string>) {
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
  vi.spyOn(SecretsService, "getSecrets").mockResolvedValue(
    Object.keys(secrets).map((name) => ({ name })),
  );
  vi.spyOn(SettingsClient.prototype, "getSecret").mockImplementation(
    async (name: string) => secrets[name],
  );
}

const gitlabMergeRequest = {
  id: 101,
  iid: 7,
  title: "Add widget export",
  web_url: "https://gitlab.example.com/acme/widgets/-/merge_requests/7",
  author: { username: "dev" },
  updated_at: "2026-11-13T10:00:00Z",
};

const originalFetch = global.fetch;
const fetchMock = vi.fn();

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([cloudBackend]);
  setActiveSelection({ backendId: cloudBackend.id });
  fetchMock.mockReset();
  vi.restoreAllMocks();
  global.fetch = fetchMock as unknown as typeof fetch;
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  fetchMock.mockReset();
  vi.restoreAllMocks();
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

  describe("self-hosted GitLab", () => {
    it("calls gitlab.com when no GitLab host is configured", async () => {
      useLocalSecrets({ GITLAB_TOKEN: "glpat-test" });
      fetchMock.mockResolvedValueOnce(mockJsonResponse([]));

      await GitProviderItemsService.listPullRequests("acme/widgets", "gitlab");

      const [url, init] = getFetchCall(fetchMock);
      expect(url).toBe(
        "https://gitlab.com/api/v4/projects/acme%2Fwidgets/merge_requests?state=opened&per_page=30",
      );
      expect(init.headers).toMatchObject({ "PRIVATE-TOKEN": "glpat-test" });
    });

    it("lists merge requests from the GITLAB_HOST instance with GITLAB_TOKEN", async () => {
      useLocalSecrets({
        GITLAB_TOKEN: "glpat-test",
        GITLAB_HOST: "gitlab.example.com",
      });
      fetchMock.mockResolvedValueOnce(mockJsonResponse([gitlabMergeRequest]));

      const items = await GitProviderItemsService.listPullRequests(
        "acme/widgets",
        "gitlab",
      );

      const [url, init] = getFetchCall(fetchMock);
      expect(url).toBe(
        "https://gitlab.example.com/api/v4/projects/acme%2Fwidgets/merge_requests?state=opened&per_page=30",
      );
      expect(init.headers).toMatchObject({ "PRIVATE-TOKEN": "glpat-test" });
      expect(items[0]).toMatchObject({
        number: 7,
        url: gitlabMergeRequest.web_url,
      });
    });

    it("lists issues from the GITLAB_HOST instance, keeping an explicit scheme and path", async () => {
      useLocalSecrets({ GITLAB_HOST: "http://git.internal:8080/gitlab/" });
      fetchMock.mockResolvedValueOnce(mockJsonResponse([]));

      await GitProviderItemsService.listIssues("acme/widgets", "gitlab");

      const [url] = getFetchCall(fetchMock);
      expect(url).toBe(
        "http://git.internal:8080/gitlab/api/v4/projects/acme%2Fwidgets/issues?state=opened&per_page=30",
      );
    });

    it("prefers the host the caller knows over the configured one", async () => {
      useLocalSecrets({ GITLAB_HOST: "gitlab.example.com" });
      fetchMock.mockResolvedValueOnce(mockJsonResponse([]));

      await GitProviderItemsService.listPullRequests(
        "acme/widgets",
        "gitlab",
        "git.other.example",
      );

      const [url] = getFetchCall(fetchMock);
      expect(url.startsWith("https://git.other.example/api/v4/")).toBe(true);
    });

    it("uses the cloud connection host passed by the caller", async () => {
      fetchMock.mockResolvedValueOnce(mockJsonResponse([]));

      await GitProviderItemsService.listIssues(
        "acme/widgets",
        "gitlab",
        "gitlab.corp.example",
      );

      const [url] = getFetchCall(fetchMock);
      expect(url.startsWith("https://gitlab.corp.example/api/v4/")).toBe(true);
    });

    it("builds list links on the given host", () => {
      expect(
        GitProviderItemsService.constructPullRequestsListUrl(
          "gitlab",
          "acme/widgets",
          "https://gitlab.example.com",
        ),
      ).toBe("https://gitlab.example.com/acme/widgets/-/merge_requests");
      expect(
        GitProviderItemsService.constructIssuesListUrl(
          "gitlab",
          "acme/widgets",
        ),
      ).toBe("https://gitlab.com/acme/widgets/-/issues");
    });

    it("detects a remote on the configured GitLab host", async () => {
      useLocalSecrets({ GITLAB_HOST: "https://GitLab.Example.com/" });

      await expect(
        GitProviderItemsService.detectConfiguredProvider("gitlab.example.com"),
      ).resolves.toBe("gitlab");
      await expect(
        GitProviderItemsService.detectConfiguredProvider("git.other.example"),
      ).resolves.toBeNull();
    });

    it("detects nothing without a configured host or on a cloud backend", async () => {
      await expect(
        GitProviderItemsService.detectConfiguredProvider("gitlab.example.com"),
      ).resolves.toBeNull();

      useLocalSecrets({});
      await expect(
        GitProviderItemsService.detectConfiguredProvider("gitlab.example.com"),
      ).resolves.toBeNull();
    });

    it.each([
      ["gitlab.example.com", "https://gitlab.example.com"],
      [" https://gitlab.example.com/ ", "https://gitlab.example.com"],
      ["http://10.0.0.5:8080/gitlab", "http://10.0.0.5:8080/gitlab"],
      ["", null],
      ["ftp://gitlab.example.com", null],
    ])("normalizes the configured host %j", (value, expected) => {
      expect(normalizeProviderBaseUrl(value)).toBe(expected);
    });
  });
});
