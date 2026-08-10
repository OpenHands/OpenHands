import { beforeEach, describe, expect, it, vi } from "vitest";
import axios from "axios";
import WorkspacesService from "#/api/workspaces-service/workspaces-service.api";

vi.mock("axios", () => ({
  default: {
    post: vi.fn(),
  },
}));

vi.mock("@openhands/typescript-client/clients", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@openhands/typescript-client/clients")>();
  return {
    ...actual,
    WorkspacesClient: class {
      listWorkspaces = vi.fn();
      addWorkspaces = vi.fn();
      deleteWorkspace = vi.fn();
      addWorkspaceParents = vi.fn();
      deleteWorkspaceParent = vi.fn();
      close = vi.fn();
    },
  };
});

vi.mock("#/api/agent-server-client-options", () => ({
  getAgentServerClientOptions: () => ({
    host: "http://localhost:18000",
    apiKey: "test",
  }),
  NoBackendAvailableError: class NoBackendAvailableError extends Error {
    constructor() {
      super("No backend is configured.");
      this.name = "NoBackendAvailableError";
    }
  },
}));

describe("WorkspacesService.cloneRepository", () => {
  beforeEach(() => {
    vi.mocked(axios.post).mockReset();
  });

  it("forwards the clone request to POST /api/workspaces/clone", async () => {
    vi.mocked(axios.post).mockResolvedValue({
      data: {
        path: "/data/workspaces/demo",
        name: "demo",
      },
    });

    const result = await WorkspacesService.cloneRepository({
      url: "https://github.com/org/demo.git",
      parentPath: "/data/workspaces",
      providerId: "github_work",
    });

    expect(axios.post).toHaveBeenCalledWith(
      "http://localhost:18000/api/workspaces/clone",
      {
        url: "https://github.com/org/demo.git",
        parentPath: "/data/workspaces",
        providerId: "github_work",
      },
      {
        timeout: 300_000,
        headers: { "X-Session-API-Key": "test" },
      },
    );
    expect(result).toEqual({ path: "/data/workspaces/demo", name: "demo" });
  });
});
