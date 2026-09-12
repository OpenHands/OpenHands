import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import {
  getCloudInstallations,
  getCloudRepositoryBranches,
  searchCloudRepositories,
} from "#/api/cloud/git-service.api";
import { server } from "#/mocks/node";

// The cloud client transport migrated from axios to the global `fetch` API,
// which this suite intercepts with the project's MSW server. Each helper
// registers a handler that records the outgoing request so the tests can
// assert on the request method and the query string the git-service functions
// ultimately issue, then responds with the supplied payload.
type CapturedRequest = { request: Request | null };

function interceptCloudGit(endpoint: string, payload: unknown): CapturedRequest {
  const captured: CapturedRequest = { request: null };
  server.use(
    http.get(`*/api/v1/git/${endpoint}/search`, ({ request }) => {
      captured.request = request;
      return HttpResponse.json(payload);
    }),
  );
  return captured;
}

function requestOf(captured: CapturedRequest): Request {
  if (!captured.request) {
    throw new Error("Expected a cloud git request to have been issued");
  }
  return captured.request;
}

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([cloudBackend]);
  setActiveSelection({ backendId: cloudBackend.id });
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("getCloudRepositoryBranches", () => {
  it("includes an empty query parameter when listing all branches so the upstream schema is satisfied", async () => {
    // Arrange
    const captured = interceptCloudGit("branches", {
      items: [],
      next_page_id: null,
    });

    // Act
    await getCloudRepositoryBranches({
      provider: "github",
      repository: "hieptl/hieptl",
    });

    // Assert
    const request = requestOf(captured);
    expect(request.method).toBe("GET");
    expect(Object.fromEntries(new URL(request.url).searchParams)).toEqual({
      provider: "github",
      repository: "hieptl/hieptl",
      limit: "30",
      query: "",
    });
  });

  it("forwards a non-empty query parameter when searching branches", async () => {
    // Arrange
    const captured = interceptCloudGit("branches", {
      items: [],
      next_page_id: null,
    });

    // Act
    await getCloudRepositoryBranches({
      provider: "github",
      repository: "hieptl/hieptl",
      query: "feature/login",
    });

    // Assert
    expect(requestOf(captured).url).toContain("query=feature%2Flogin");
  });

  it("forwards branch pagination and normalizes absent response fields", async () => {
    const captured = interceptCloudGit("branches", null);
    await expect(
      getCloudRepositoryBranches({
        provider: "github",
        repository: "owner/repo",
        query: "feature",
        pageId: "next",
        limit: 4,
      }),
    ).resolves.toEqual({ items: [], next_page_id: null });
    expect(requestOf(captured).url).toContain(
      "provider=github&repository=owner%2Frepo&limit=4&query=feature&page_id=next",
    );
  });
});

describe("cloud repository and installation searches", () => {
  it("forwards every repository search option and returns the page", async () => {
    const page = {
      items: [{ id: "1", full_name: "owner/repo" }],
      next_page_id: "next",
    };
    const captured = interceptCloudGit("repositories", page);
    await expect(
      searchCloudRepositories({
        provider: "github",
        query: "repo",
        pageId: "page",
        installationId: "install",
        limit: 5,
      }),
    ).resolves.toEqual(page);
    expect(requestOf(captured).url).toContain(
      "provider=github&limit=5&query=repo&page_id=page&installation_id=install",
    );
  });

  it("uses repository defaults and normalizes an absent response", async () => {
    const captured = interceptCloudGit("repositories", null);
    await expect(
      searchCloudRepositories({ provider: "github" }),
    ).resolves.toEqual({ items: [], next_page_id: null });
    const request = requestOf(captured);
    expect(request.method).toBe("GET");
    expect(Object.fromEntries(new URL(request.url).searchParams)).toEqual({
      provider: "github",
      limit: "100",
    });
  });

  it("forwards installation pagination and returns the page", async () => {
    const page = { items: ["installation"], next_page_id: "next" };
    const captured = interceptCloudGit("installations", page);
    await expect(
      getCloudInstallations({ provider: "gitlab", pageId: "page", limit: 6 }),
    ).resolves.toEqual(page);
    expect(requestOf(captured).url).toContain(
      "provider=gitlab&limit=6&page_id=page",
    );
  });

  it("uses installation defaults and normalizes an absent response", async () => {
    const captured = interceptCloudGit("installations", null);
    await expect(
      getCloudInstallations({ provider: "github" }),
    ).resolves.toEqual({ items: [], next_page_id: null });
    const request = requestOf(captured);
    expect(request.method).toBe("GET");
    expect(Object.fromEntries(new URL(request.url).searchParams)).toEqual({
      provider: "github",
      limit: "100",
    });
  });

  it("rejects direct cloud git calls for a local backend", async () => {
    setRegisteredBackends([
      {
        id: "local",
        name: "Local",
        host: "http://localhost:8000",
        apiKey: "key",
        kind: "local",
      },
    ]);
    setActiveSelection({ backendId: "local" });
    await expect(getCloudInstallations({ provider: "github" })).rejects.toThrow(
      "Cloud git call requires a cloud backend.",
    );
  });
});
