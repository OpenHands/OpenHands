import { describe, expect, it, vi } from "vitest";
import WorkspacesService from "#/api/workspaces-service/workspaces-service.api";

const cloneRepository = vi.fn();

vi.mock("@openhands/typescript-client/clients", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@openhands/typescript-client/clients")>();
  return {
    ...actual,
    WorkspacesClient: class {
      cloneRepository = cloneRepository;
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
}));

describe("WorkspacesService.cloneRepository", () => {
  it("forwards the clone request to WorkspacesClient", async () => {
    cloneRepository.mockResolvedValue({
      path: "/data/workspaces/demo",
      name: "demo",
    });

    const result = await WorkspacesService.cloneRepository({
      url: "https://github.com/org/demo.git",
      parentPath: "/data/workspaces",
      providerId: "github_work",
    });

    expect(cloneRepository).toHaveBeenCalledWith({
      url: "https://github.com/org/demo.git",
      parentPath: "/data/workspaces",
      providerId: "github_work",
    });
    expect(result).toEqual({ path: "/data/workspaces/demo", name: "demo" });
  });
});
