import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import WorkspacesService from "#/api/workspaces-service/workspaces-service.api";

const {
  mockListWorkspaces,
  mockAddWorkspaces,
  mockDeleteWorkspace,
  mockAddWorkspaceParents,
  mockDeleteWorkspaceParent,
} = vi.hoisted(() => ({
  mockListWorkspaces: vi.fn(),
  mockAddWorkspaces: vi.fn(),
  mockDeleteWorkspace: vi.fn(),
  mockAddWorkspaceParents: vi.fn(),
  mockDeleteWorkspaceParent: vi.fn(),
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  WorkspacesClient: vi.fn(function WorkspacesClientMock() {
    return {
      listWorkspaces: mockListWorkspaces,
      addWorkspaces: mockAddWorkspaces,
      deleteWorkspace: mockDeleteWorkspace,
      addWorkspaceParents: mockAddWorkspaceParents,
      deleteWorkspaceParent: mockDeleteWorkspaceParent,
    };
  }),
}));

const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://127.0.0.1:8000",
  apiKey: "",
  kind: "local",
};

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
  mockListWorkspaces.mockReset();
  mockAddWorkspaces.mockReset();
  mockDeleteWorkspace.mockReset();
  mockAddWorkspaceParents.mockReset();
  mockDeleteWorkspaceParent.mockReset();
});

afterEach(() => {
  __resetActiveStoreForTests();
});

describe("WorkspacesService", () => {
  it("listWorkspaces delegates to WorkspacesClient", async () => {
    // Arrange
    const body = {
      workspaces: [{ id: "/a", name: "a", path: "/a" }],
      workspaceParents: [{ id: "/p", name: "p", path: "/p" }],
    };
    mockListWorkspaces.mockResolvedValue(body);

    // Act
    const result = await WorkspacesService.listWorkspaces();

    // Assert
    expect(mockListWorkspaces).toHaveBeenCalledWith();
    expect(result).toEqual(body);
  });

  it("propagates the typed old-server error from WorkspacesClient", async () => {
    // Arrange
    const error = {
      code: "AGENT_SERVER_VERSION_TOO_OLD",
      feature: "workspaces",
      requiredVersion: "1.23.0",
      actualVersion: "1.22.1",
    };
    mockListWorkspaces.mockRejectedValue(error);

    // Act + Assert
    await expect(WorkspacesService.listWorkspaces()).rejects.toBe(error);
  });

  it("addWorkspaces delegates to WorkspacesClient", async () => {
    // Arrange
    mockAddWorkspaces.mockResolvedValue({
      workspaces: [],
      workspaceParents: [],
    });
    const items = [{ id: "/a", name: "a", path: "/a", parentPath: "/p" }];

    // Act
    await WorkspacesService.addWorkspaces(items);

    // Assert
    expect(mockAddWorkspaces).toHaveBeenCalledWith(items);
  });

  it("addWorkspaceParents delegates to WorkspacesClient", async () => {
    // Arrange
    mockAddWorkspaceParents.mockResolvedValue({
      workspaces: [],
      workspaceParents: [],
    });
    const parents = [{ id: "/p", name: "p", path: "/p" }];

    // Act
    await WorkspacesService.addWorkspaceParents(parents);

    // Assert
    expect(mockAddWorkspaceParents).toHaveBeenCalledWith(parents);
  });

  it("removeWorkspace delegates to WorkspacesClient", async () => {
    // Arrange
    mockDeleteWorkspace.mockResolvedValue({ deleted: true });

    // Act
    await WorkspacesService.removeWorkspace("/Users/me/dev/repo 1");

    // Assert
    expect(mockDeleteWorkspace).toHaveBeenCalledWith("/Users/me/dev/repo 1");
  });

  it("removeWorkspaceParent delegates to WorkspacesClient", async () => {
    // Arrange
    mockDeleteWorkspaceParent.mockResolvedValue({ deleted: true });

    // Act
    await WorkspacesService.removeWorkspaceParent("/parents/root");

    // Assert
    expect(mockDeleteWorkspaceParent).toHaveBeenCalledWith("/parents/root");
  });
});
