import { describe, expect, vi, beforeEach, it, afterEach } from "vitest";
import GitService from "#/api/git-service/git-service.api";
import * as cloudGitService from "#/api/cloud/git-service.api";
import * as activeStore from "#/api/backend-registry/active-store";

const workspaceMocks = vi.hoisted(() => ({
  gitChanges: vi.fn(),
  gitDiff: vi.fn(),
  resolveWorkingDir: vi.fn(),
  clientOptions: vi.fn(),
  mapStatus: vi.fn(),
}));

vi.mock("#/api/cloud/git-service.api", () => ({
  searchCloudRepositories: vi.fn(),
  getCloudInstallations: vi.fn(),
  getCloudRepositoryBranches: vi.fn(),
}));

vi.mock("#/api/backend-registry/active-store", () => ({
  getActiveBackend: vi.fn(),
}));

vi.mock("@openhands/typescript-client/workspace/remote-workspace", () => ({
  RemoteWorkspace: vi.fn(function RemoteWorkspaceMock() {
    return {
      gitChanges: workspaceMocks.gitChanges,
      gitDiff: workspaceMocks.gitDiff,
    };
  }),
}));

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: {
      resolveConversationWorkingDir: workspaceMocks.resolveWorkingDir,
    },
  }),
);

vi.mock("#/api/agent-server-client-options", () => ({
  getAgentServerClientOptions: workspaceMocks.clientOptions,
}));

vi.mock("#/utils/git-status-mapper", () => ({
  mapAnyGitStatusToClientStatus: workspaceMocks.mapStatus,
}));

const mockSearchCloudRepositories = vi.mocked(
  cloudGitService.searchCloudRepositories,
);
const mockGetCloudInstallations = vi.mocked(
  cloudGitService.getCloudInstallations,
);
const mockGetCloudRepositoryBranches = vi.mocked(
  cloudGitService.getCloudRepositoryBranches,
);
const mockGetActiveBackend = vi.mocked(activeStore.getActiveBackend);

const cloudActive = () =>
  mockGetActiveBackend.mockReturnValue({
    backend: {
      kind: "cloud",
      id: "test",
      name: "Test",
      host: "https://example.com",
      apiKey: "test-key",
    },
    orgId: "org-1",
  });

const localActive = () =>
  mockGetActiveBackend.mockReturnValue({
    backend: {
      kind: "local",
      id: "test",
      name: "Test",
      host: "http://localhost",
      apiKey: "test-key",
    },
    orgId: null,
  });

describe("GitService", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    workspaceMocks.resolveWorkingDir.mockResolvedValue("/workspace/project");
    workspaceMocks.clientOptions.mockReturnValue({ host: "http://localhost" });
    workspaceMocks.mapStatus.mockImplementation((status) => `mapped:${status}`);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe("invalid provider guards", () => {
    const invalidProviders = [
      { value: null, name: "null" },
      { value: undefined, name: "undefined" },
      { value: "", name: "empty string" },
      { value: "undefined", name: '"undefined" string' },
      { value: "null", name: '"null" string' },
    ];

    describe("searchGitRepositories", () => {
      it.each(invalidProviders)(
        "should return empty results when provider is $name (cloud mode)",
        async ({ value }) => {
          cloudActive();

          const result = await GitService.searchGitRepositories(
            "test query",
            value as string,
          );

          expect(result).toEqual({ items: [], next_page_id: null });
          expect(mockSearchCloudRepositories).not.toHaveBeenCalled();
        },
      );
    });

    describe("retrieveUserGitRepositories", () => {
      it.each(invalidProviders)(
        "should return empty results when provider is $name",
        async ({ value }) => {
          cloudActive();

          const result = await GitService.retrieveUserGitRepositories(
            value as string,
          );

          expect(result).toEqual({ items: [], next_page_id: null });
          expect(mockSearchCloudRepositories).not.toHaveBeenCalled();
        },
      );
    });

    describe("retrieveInstallationRepositories", () => {
      it.each(invalidProviders)(
        "should return empty results when provider is $name",
        async ({ value }) => {
          cloudActive();

          const result = await GitService.retrieveInstallationRepositories(
            value as string,
            0,
            ["installation-1"],
          );

          expect(result).toEqual({ items: [], next_page_id: null });
          expect(mockSearchCloudRepositories).not.toHaveBeenCalled();
        },
      );
    });

    describe("getUserInstallations", () => {
      it.each(invalidProviders)(
        "should return empty results when provider is $name",
        async ({ value }) => {
          cloudActive();

          const result = await GitService.getUserInstallations(value as string);

          expect(result).toEqual({ items: [], next_page_id: null });
          expect(mockGetCloudInstallations).not.toHaveBeenCalled();
        },
      );
    });

    describe("getRepositoryBranches", () => {
      it.each(invalidProviders)(
        "should return empty results when provider is $name",
        async ({ value }) => {
          cloudActive();

          const result = await GitService.getRepositoryBranches(
            "owner/repo",
            value as string,
          );

          expect(result).toEqual({ items: [], next_page_id: null });
          expect(mockGetCloudRepositoryBranches).not.toHaveBeenCalled();
        },
      );
    });

    describe("searchRepositoryBranches", () => {
      it.each(invalidProviders)(
        "should return empty results when provider is $name",
        async ({ value }) => {
          cloudActive();

          const result = await GitService.searchRepositoryBranches(
            "owner/repo",
            value as string,
            "main",
          );

          expect(result).toEqual({ items: [], next_page_id: null });
          expect(mockGetCloudRepositoryBranches).not.toHaveBeenCalled();
        },
      );
    });
  });

  describe("valid provider behavior", () => {
    it("should call cloud API when provider is valid and cloud is active", async () => {
      cloudActive();
      mockSearchCloudRepositories.mockResolvedValue({
        items: [
          {
            id: "1",
            full_name: "owner/repo",
            git_provider: "github",
            is_public: true,
          },
        ],
        next_page_id: null,
      });

      const result = await GitService.searchGitRepositories("test", "github");

      expect(mockSearchCloudRepositories).toHaveBeenCalledWith({
        provider: "github",
        query: "test",
        limit: 100,
        pageId: undefined,
        installationId: undefined,
      });
      expect(result.items).toHaveLength(1);
    });

    it("should short-circuit to empty results when provider is valid but local backend is active", async () => {
      localActive();

      const result = await GitService.searchGitRepositories("test", "github");

      expect(result).toEqual({ items: [], next_page_id: null });
      expect(mockSearchCloudRepositories).not.toHaveBeenCalled();
    });

    it("routes every supported cloud lookup with normalized parameters", async () => {
      cloudActive();
      const repositories = { items: [], next_page_id: "repos-next" };
      const branches = { items: [], next_page_id: "branches-next" };
      const installations = { items: [], next_page_id: "installs-next" };
      mockSearchCloudRepositories.mockResolvedValue(repositories);
      mockGetCloudRepositoryBranches.mockResolvedValue(branches);
      mockGetCloudInstallations.mockResolvedValue(installations);

      await expect(
        GitService.searchGitRepositories("", "github", 5, "page", "install"),
      ).resolves.toBe(repositories);
      expect(mockSearchCloudRepositories).toHaveBeenLastCalledWith({
        provider: "github",
        query: undefined,
        limit: 5,
        pageId: "page",
        installationId: "install",
      });

      await expect(
        GitService.retrieveUserGitRepositories("github", "next", 6, "install"),
      ).resolves.toBe(repositories);
      expect(mockSearchCloudRepositories).toHaveBeenLastCalledWith({
        provider: "github",
        limit: 6,
        pageId: "next",
        installationId: "install",
      });

      await expect(
        GitService.retrieveInstallationRepositories(
          "github",
          1,
          ["first", "second"],
          "next",
          7,
        ),
      ).resolves.toBe(repositories);
      expect(mockSearchCloudRepositories).toHaveBeenLastCalledWith({
        provider: "github",
        installationId: "second",
        limit: 7,
        pageId: "next",
      });
      await expect(
        GitService.retrieveInstallationRepositories("github", 5, ["first"]),
      ).resolves.toEqual({ items: [], next_page_id: null });

      await expect(
        GitService.getRepositoryBranches("owner/repo", "github", "", "next", 8),
      ).resolves.toBe(branches);
      expect(mockGetCloudRepositoryBranches).toHaveBeenLastCalledWith({
        provider: "github",
        repository: "owner/repo",
        query: undefined,
        pageId: "next",
        limit: 8,
      });

      await expect(
        GitService.searchRepositoryBranches(
          "owner/repo",
          "github",
          "feature",
          "next",
          9,
        ),
      ).resolves.toBe(branches);
      expect(mockGetCloudRepositoryBranches).toHaveBeenLastCalledWith({
        provider: "github",
        repository: "owner/repo",
        query: "feature",
        pageId: "next",
        limit: 9,
      });

      await expect(
        GitService.getUserInstallations("github", "next", 10),
      ).resolves.toBe(installations);
      expect(mockGetCloudInstallations).toHaveBeenCalledWith({
        provider: "github",
        pageId: "next",
        limit: 10,
      });
    });

    it("short-circuits every cloud-only lookup for a local backend", async () => {
      localActive();
      await expect(
        GitService.retrieveUserGitRepositories("github"),
      ).resolves.toEqual({ items: [], next_page_id: null });
      await expect(
        GitService.retrieveInstallationRepositories("github", 0, ["install"]),
      ).resolves.toEqual({ items: [], next_page_id: null });
      await expect(
        GitService.getRepositoryBranches("owner/repo", "github"),
      ).resolves.toEqual({ items: [], next_page_id: null });
      await expect(
        GitService.searchRepositoryBranches("owner/repo", "github", "main"),
      ).resolves.toEqual({ items: [], next_page_id: null });
      await expect(GitService.getUserInstallations("github")).resolves.toEqual({
        items: [],
        next_page_id: null,
      });
    });

    it("maps workspace git changes using the SDK default revision", async () => {
      workspaceMocks.gitChanges.mockResolvedValue([
        { path: "src/a.ts", status: "M" },
        { path: "src/b.ts", status: 7 },
      ]);
      await expect(GitService.getGitChanges("conversation-1")).resolves.toEqual(
        [
          { path: "src/a.ts", status: "mapped:M" },
          { path: "src/b.ts", status: "mapped:7" },
        ],
      );
      expect(workspaceMocks.clientOptions).toHaveBeenCalledWith({
        workingDir: "/workspace/project",
      });
      expect(workspaceMocks.gitChanges).toHaveBeenCalledWith(
        "/workspace/project",
      );
    });

    it.each([
      [
        { modified: "new", original: "old" },
        { modified: "new", original: "old" },
      ],
      [{}, { modified: "", original: "" }],
    ])(
      "normalizes workspace diff content using the SDK default revision",
      async (diff, expected) => {
        workspaceMocks.gitDiff.mockResolvedValue(diff);
        await expect(
          GitService.getGitChangeDiff("ignored", "src/a.ts"),
        ).resolves.toEqual(expected);
        expect(workspaceMocks.gitDiff).toHaveBeenCalledWith("src/a.ts");
      },
    );

    it("initializes fresh provider guards and empty result pages", async () => {
      vi.resetModules();

      try {
        const freshCloud = await import("#/api/cloud/git-service.api");
        const freshActiveStore =
          await import("#/api/backend-registry/active-store");
        const { default: FreshGitService } =
          await import("#/api/git-service/git-service.api");
        const freshSearch = vi.mocked(freshCloud.searchCloudRepositories);
        const freshBranches = vi.mocked(freshCloud.getCloudRepositoryBranches);
        const freshInstallations = vi.mocked(freshCloud.getCloudInstallations);
        vi.mocked(freshActiveStore.getActiveBackend).mockReturnValue({
          backend: {
            kind: "cloud",
            id: "fresh",
            name: "Fresh",
            host: "https://example.com",
            apiKey: "key",
          },
          orgId: "org-1",
        });

        const repositoryPage = { items: [], next_page_id: "fresh-repo" };
        const branchPage = { items: [], next_page_id: "fresh-branch" };
        freshSearch.mockResolvedValue(repositoryPage);
        freshBranches.mockResolvedValue(branchPage);

        await expect(
          FreshGitService.searchGitRepositories("query", "github"),
        ).resolves.toBe(repositoryPage);
        expect(freshSearch).toHaveBeenCalledWith(
          expect.objectContaining({ provider: "github" }),
        );

        await expect(
          FreshGitService.getRepositoryBranches("owner/repo", "github"),
        ).resolves.toBe(branchPage);
        expect(freshBranches).toHaveBeenCalledWith({
          provider: "github",
          repository: "owner/repo",
          query: undefined,
          pageId: undefined,
          limit: 30,
        });

        vi.clearAllMocks();
        await expect(
          FreshGitService.searchGitRepositories("query", "undefined"),
        ).resolves.toEqual({ items: [], next_page_id: null });
        await expect(
          FreshGitService.getRepositoryBranches("owner/repo", "undefined"),
        ).resolves.toEqual({ items: [], next_page_id: null });
        await expect(
          FreshGitService.getUserInstallations("undefined"),
        ).resolves.toEqual({ items: [], next_page_id: null });
        expect(freshSearch).not.toHaveBeenCalled();
        expect(freshBranches).not.toHaveBeenCalled();
        expect(freshInstallations).not.toHaveBeenCalled();
      } finally {
        vi.resetModules();
      }
    });
  });
});
