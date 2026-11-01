import { describe, expect, it, vi } from "vitest";
import { server } from "#/mocks/node";
import type { Branch, GitRepository } from "#/types/git";

const API_BASE = "http://localhost:3000";

const getJson = async <T>(path: string) => {
  const response = await fetch(`${API_BASE}${path}`);
  return {
    body: (await response.json()) as T,
    headers: response.headers,
    status: response.status,
  };
};

describe("mock git browsing", () => {
  it("requires a provider before listing or searching repositories", async () => {
    const listing = await getJson<string>("/api/user/repositories");
    const search = await getJson<string>("/api/user/search/repositories");

    expect(listing).toMatchObject({
      body: "Git provider token required. (such as GitHub).",
      status: 401,
    });
    expect(search).toMatchObject({
      body: "Git provider token required.",
      status: 401,
    });
  });

  it("lists the newest repositories first with navigable pagination", async () => {
    const firstPage = await getJson<GitRepository[]>(
      "/api/user/repositories?selected_provider=github&per_page=20",
    );
    const secondPage = await getJson<GitRepository[]>(
      "/api/user/repositories?selected_provider=github&page=2&per_page=20",
    );
    const lastPage = await getJson<GitRepository[]>(
      "/api/user/repositories?selected_provider=github&page=6&per_page=20",
    );

    expect(firstPage.status).toBe(200);
    expect(firstPage.body).toHaveLength(20);
    expect(
      firstPage.body.every(
        (repository) => repository.git_provider === "github",
      ),
    ).toBe(true);
    expect(
      firstPage.body.map(({ pushed_at }) => Date.parse(pushed_at ?? "")),
    ).toEqual(
      [...firstPage.body]
        .map(({ pushed_at }) => Date.parse(pushed_at ?? ""))
        .sort((a, b) => b - a),
    );
    expect(firstPage.body[0]?.link_header).toBe(
      '</api/user/repositories?page=2&per_page=20>; rel="next", </api/user/repositories?page=6&per_page=20>; rel="last", </api/user/repositories?page=1&per_page=20>; rel="first"',
    );
    expect(secondPage.body[0]?.link_header).toBe(
      '</api/user/repositories?page=1&per_page=20>; rel="prev", </api/user/repositories?page=3&per_page=20>; rel="next", </api/user/repositories?page=6&per_page=20>; rel="last", </api/user/repositories?page=1&per_page=20>; rel="first"',
    );
    expect(lastPage.body[0]?.link_header).toBe(
      '</api/user/repositories?page=5&per_page=20>; rel="prev", </api/user/repositories?page=6&per_page=20>; rel="last", </api/user/repositories?page=1&per_page=20>; rel="first"',
    );
  });

  it("supports star sorting, alternative providers, and unpaginated responses", async () => {
    const starred = await getJson<GitRepository[]>(
      "/api/user/repositories?selected_provider=gitlab&sort=stars&per_page=200",
    );
    const originalOrder = await getJson<GitRepository[]>(
      "/api/user/repositories?selected_provider=bitbucket&sort=name&per_page=200",
    );
    const unsupported = await getJson<GitRepository[]>(
      "/api/user/repositories?selected_provider=unknown",
    );

    expect(
      starred.body.map(({ stargazers_count }) => stargazers_count),
    ).toEqual(
      [...starred.body]
        .map(({ stargazers_count }) => stargazers_count)
        .sort((a, b) => (b ?? 0) - (a ?? 0)),
    );
    expect(starred.body[0]?.link_header).toBeUndefined();
    expect(originalOrder.body.map(({ id }) => id)).toEqual(
      Array.from({ length: 120 }, (_, index) => `${index + 1}`),
    );
    expect(unsupported.body).toEqual([]);
  });

  it("limits GitHub repositories to the selected installation", async () => {
    const firstInstallation = await getJson<GitRepository[]>(
      "/api/user/repositories?selected_provider=github&installation_id=invalid&sort=name&per_page=30",
    );
    const secondInstallation = await getJson<GitRepository[]>(
      "/api/user/repositories?selected_provider=github&installation_id=1&sort=name&per_page=30",
    );
    const ignoredForGitLab = await getJson<GitRepository[]>(
      "/api/user/repositories?selected_provider=gitlab&installation_id=1&sort=name&per_page=200",
    );

    expect(firstInstallation.body.map(({ id }) => id)).toEqual(
      Array.from({ length: 20 }, (_, index) => `${index + 1}`),
    );
    expect(secondInstallation.body.map(({ id }) => id)).toEqual(
      Array.from({ length: 20 }, (_, index) => `${index + 21}`),
    );
    expect(ignoredForGitLab.body).toHaveLength(120);
  });

  it("returns an empty page without attaching pagination metadata", async () => {
    const response = await getJson<GitRepository[]>(
      "/api/user/repositories?selected_provider=github&page=99&per_page=30",
    );

    expect(response.body).toEqual([]);
  });

  it("uses the public default page sizes for repository lists and searches", async () => {
    const listing = await getJson<GitRepository[]>(
      "/api/user/repositories?selected_provider=github&sort=name",
    );
    const search = await getJson<GitRepository[]>(
      "/api/user/search/repositories?selected_provider=github&query=repo&sort=name",
    );

    expect(listing.body).toHaveLength(30);
    expect(listing.body[0]?.link_header).toContain("per_page=30");
    expect(search.body).toHaveLength(5);
  });

  it("searches repositories case-insensitively and honors ordering and limits", async () => {
    const descending = await getJson<GitRepository[]>(
      "/api/user/search/repositories?selected_provider=github&query=REPO-&per_page=8",
    );
    const ascending = await getJson<GitRepository[]>(
      "/api/user/search/repositories?selected_provider=github&query=repo-&per_page=8&sort=stars&order=asc",
    );
    const originalOrder = await getJson<GitRepository[]>(
      "/api/user/search/repositories?selected_provider=github&query=repo-1&per_page=50&sort=name",
    );
    const unsupported = await getJson<GitRepository[]>(
      "/api/user/search/repositories?selected_provider=unknown&query=repo",
    );

    expect(descending.body).toHaveLength(8);
    expect(
      descending.body.map(({ stargazers_count }) => stargazers_count),
    ).toEqual(
      [...descending.body]
        .map(({ stargazers_count }) => stargazers_count)
        .sort((a, b) => (b ?? 0) - (a ?? 0)),
    );
    expect(
      ascending.body.map(({ stargazers_count }) => stargazers_count),
    ).toEqual(
      [...ascending.body]
        .map(({ stargazers_count }) => stargazers_count)
        .sort((a, b) => (a ?? 0) - (b ?? 0)),
    );
    expect(originalOrder.body.map(({ id }) => id)).toEqual([
      "1",
      "10",
      "11",
      "12",
      "13",
      "14",
      "15",
      "16",
      "17",
      "18",
      "19",
      "100",
      "101",
      "102",
      "103",
      "104",
      "105",
      "106",
      "107",
      "108",
      "109",
      "110",
      "111",
      "112",
      "113",
      "114",
      "115",
      "116",
      "117",
      "118",
      "119",
      "120",
    ]);
    expect(unsupported.body).toEqual([]);
  });

  it("validates repository names and paginates branch listings", async () => {
    const missing = await getJson<string>("/api/user/repository/branches");
    const firstPage = await getJson<{
      branches: Branch[];
      current_page: number;
      has_next_page: boolean;
      per_page: number;
      total_count: number;
    }>("/api/user/repository/branches?repository=user/repo");
    const lastPage = await getJson<{
      branches: Branch[];
      current_page: number;
      has_next_page: boolean;
      per_page: number;
      total_count: number;
    }>("/api/user/repository/branches?repository=user/repo&page=3&per_page=10");
    const exactPage = await getJson<{
      branches: Branch[];
      has_next_page: boolean;
    }>("/api/user/repository/branches?repository=user/repo&per_page=25");

    expect(missing).toMatchObject({
      body: "Repository parameter is required",
      status: 400,
    });
    expect(firstPage.body).toMatchObject({
      current_page: 1,
      has_next_page: false,
      per_page: 30,
      total_count: 25,
    });
    expect(firstPage.body.branches).toHaveLength(25);
    expect(firstPage.body.branches[0]).toMatchObject({
      name: "main",
      protected: true,
    });
    expect(firstPage.body.branches[1]).toMatchObject({
      name: "develop",
      protected: false,
    });
    expect(firstPage.body.branches[2]?.name).toBe("feature/branch-2");
    expect(lastPage.body).toMatchObject({
      current_page: 3,
      has_next_page: false,
      per_page: 10,
      total_count: 25,
    });
    expect(lastPage.body.branches).toHaveLength(5);
    expect(exactPage.body.branches).toHaveLength(25);
    expect(exactPage.body.has_next_page).toBe(false);
  });

  it("reports when another branch page exists", async () => {
    const response = await getJson<{
      branches: Branch[];
      has_next_page: boolean;
    }>("/api/user/repository/branches?repository=user/repo&page=1&per_page=10");

    expect(response.body.branches).toHaveLength(10);
    expect(response.body.has_next_page).toBe(true);
  });

  it("validates and searches branch names case-insensitively", async () => {
    const missing = await getJson<string>("/api/user/search/branches");
    const defaults = await getJson<Branch[]>(
      "/api/user/search/branches?repository=user/repo",
    );
    const featureMatches = await getJson<Branch[]>(
      "/api/user/search/branches?repository=user/repo&query=FEATURE&per_page=2",
    );

    expect(missing).toMatchObject({
      body: "Repository parameter is required",
      status: 400,
    });
    expect(defaults.body).toHaveLength(25);
    expect(featureMatches.body).toHaveLength(2);
    expect(
      featureMatches.body.every(({ name }) => name.startsWith("feature/")),
    ).toBe(true);
  });

  it("serves representative workspace changes and unified diff content", async () => {
    const changes =
      await getJson<Array<{ path: string; status: string }>>(
        "/api/git/changes",
      );
    const diff = await getJson<{ modified: string; original: string }>(
      "/api/git/diff",
    );

    expect(changes.body).toEqual([
      { path: "src/components/hello.tsx", status: "UPDATED" },
      { path: "src/utils/new-helper.ts", status: "ADDED" },
      { path: "src/old-module.py", status: "DELETED" },
    ]);
    expect(diff.body).toEqual({
      original: 'def greet(name):\n    return f"Hello, {name}!"\n',
      modified:
        'def greet(name: str) -> str:\n    return f"Hello, {name}! Welcome."\n',
    });
  });

  it("keeps star ordering stable when generated provider metadata has no score", async () => {
    let randomCall = 0;
    const random = vi.spyOn(Math, "random").mockImplementation(() => {
      const repositoryField = randomCall % 3;
      randomCall += 1;
      return randomCall <= 1_080 && repositoryField === 1 ? Number.NaN : 0.5;
    });

    vi.resetModules();
    const { GIT_REPOSITORY_HANDLERS } =
      await import("#/mocks/git-repository-handlers");
    random.mockRestore();
    server.resetHandlers(...GIT_REPOSITORY_HANDLERS);

    const listing = await getJson<GitRepository[]>(
      "/api/user/repositories?selected_provider=github&sort=stars&per_page=3",
    );
    const search = await getJson<GitRepository[]>(
      "/api/user/search/repositories?selected_provider=github&sort=stars&per_page=3",
    );

    expect(listing.status).toBe(200);
    expect(
      listing.body.map(({ stargazers_count }) => stargazers_count),
    ).toEqual([null, null, null]);
    expect(search.status).toBe(200);
    expect(search.body.map(({ stargazers_count }) => stargazers_count)).toEqual(
      [null, null, null],
    );
  });

  it("serves deterministic generated metadata through every registered route", async () => {
    const fixedNow = Date.parse("2025-01-15T12:00:00.000Z");
    const random = vi.spyOn(Math, "random").mockReturnValue(0.3);
    const now = vi.spyOn(Date, "now").mockReturnValue(fixedNow);

    vi.resetModules();
    const { GIT_REPOSITORY_HANDLERS } =
      await import("#/mocks/git-repository-handlers");
    random.mockRestore();
    now.mockRestore();
    server.resetHandlers(...GIT_REPOSITORY_HANDLERS);

    const github = await getJson<GitRepository[]>(
      "/api/user/repositories?selected_provider=github&sort=name&per_page=120",
    );
    const gitlab = await getJson<GitRepository[]>(
      "/api/user/repositories?selected_provider=gitlab&sort=name&per_page=120",
    );
    const bitbucket = await getJson<GitRepository[]>(
      "/api/user/repositories?selected_provider=bitbucket&sort=name&per_page=120",
    );
    const search = await getJson<GitRepository[]>(
      "/api/user/search/repositories?selected_provider=github&query=repo-1&sort=name&per_page=1",
    );
    const branches = await getJson<{ branches: Branch[] }>(
      "/api/user/repository/branches?repository=user/repo",
    );
    const branchSearch = await getJson<Branch[]>(
      "/api/user/search/branches?repository=user/repo&query=develop",
    );
    const changes =
      await getJson<Array<{ path: string; status: string }>>(
        "/api/git/changes",
      );
    const diff = await getJson<{ modified: string; original: string }>(
      "/api/git/diff",
    );

    const pushedAt = new Date(
      fixedNow - 0.3 * 90 * 24 * 60 * 60 * 1000,
    ).toISOString();
    const lastPushDate = new Date(
      fixedNow - 0.3 * 7 * 24 * 60 * 60 * 1000,
    ).toISOString();

    expect(github.body).toHaveLength(120);
    expect(github.body[0]).toEqual({
      id: "1",
      full_name: "user/repo-1",
      git_provider: "github",
      is_public: false,
      stargazers_count: 300,
      pushed_at: pushedAt,
    });
    expect(github.body[119]?.id).toBe("120");
    expect(gitlab.body[0]?.git_provider).toBe("gitlab");
    expect(bitbucket.body[0]?.git_provider).toBe("bitbucket");
    expect(search.body[0]?.full_name).toBe("user/repo-1");

    expect(branches.body.branches).toHaveLength(25);
    expect(branches.body.branches.slice(0, 3)).toEqual([
      {
        name: "main",
        commit_sha: "abc123000",
        protected: true,
        last_push_date: lastPushDate,
      },
      {
        name: "develop",
        commit_sha: "abc123001",
        protected: false,
        last_push_date: lastPushDate,
      },
      {
        name: "feature/branch-2",
        commit_sha: "abc123002",
        protected: false,
        last_push_date: lastPushDate,
      },
    ]);
    expect(branchSearch.body).toEqual([branches.body.branches[1]]);
    expect(changes.body).toEqual([
      { path: "src/components/hello.tsx", status: "UPDATED" },
      { path: "src/utils/new-helper.ts", status: "ADDED" },
      { path: "src/old-module.py", status: "DELETED" },
    ]);
    expect(diff.body).toEqual({
      original: 'def greet(name):\n    return f"Hello, {name}!"\n',
      modified:
        'def greet(name: str) -> str:\n    return f"Hello, {name}! Welcome."\n',
    });
  });
});
