import { afterEach, describe, expect, it } from "vitest";

import WorkspacesService from "#/api/workspaces-service/workspaces-service.api";
import { resetMockWorkspaces } from "#/mocks/handlers";

describe("mock workspaces handlers", () => {
  afterEach(() => {
    resetMockWorkspaces();
  });

  it("starts with an empty workspaces list", async () => {
    const response = await WorkspacesService.listWorkspaces();
    expect(response.workspaces).toEqual([]);
    expect(response.workspaceParents).toEqual([]);
  });

  it("persists added workspaces across list calls", async () => {
    await WorkspacesService.addWorkspaces([
      { id: "w1", name: "Project", path: "/workspace/project" },
    ]);

    const response = await WorkspacesService.listWorkspaces();
    expect(response.workspaces).toEqual([
      { id: "w1", name: "Project", path: "/workspace/project" },
    ]);
  });

  it("upserts a workspace when path already exists", async () => {
    await WorkspacesService.addWorkspaces([
      { id: "w1", name: "Old", path: "/workspace/project" },
    ]);
    await WorkspacesService.addWorkspaces([
      { id: "w2", name: "New", path: "/workspace/project" },
    ]);
    const response = await WorkspacesService.listWorkspaces();
    expect(response.workspaces).toHaveLength(1);
    expect(response.workspaces[0].name).toBe("New");
  });

  it("removes a workspace by path", async () => {
    await WorkspacesService.addWorkspaces([
      { id: "w1", name: "Project", path: "/workspace/project" },
      { id: "w2", name: "Other", path: "/workspace/other" },
    ]);

    await WorkspacesService.removeWorkspace("/workspace/project");

    const response = await WorkspacesService.listWorkspaces();
    expect(response.workspaces.map((w) => w.path)).toEqual([
      "/workspace/other",
    ]);
  });

  it("persists workspace parents and removes them by path", async () => {
    await WorkspacesService.addWorkspaceParents([
      { id: "p1", name: "Repos", path: "/workspace/repos" },
    ]);

    const afterAdd = await WorkspacesService.listWorkspaces();
    expect(afterAdd.workspaceParents).toEqual([
      { id: "p1", name: "Repos", path: "/workspace/repos" },
    ]);

    await WorkspacesService.removeWorkspaceParent("/workspace/repos");

    const afterRemove = await WorkspacesService.listWorkspaces();
    expect(afterRemove.workspaceParents).toEqual([]);
  });
});
