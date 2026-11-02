import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";

import { server } from "#/mocks/node";

const unhandledWorkspacesRequest = () =>
  HttpResponse.json({ error: "Unhandled workspaces request" }, { status: 599 });

const UNHANDLED_WORKSPACES_REQUESTS = [
  http.all("*/api/workspaces", unhandledWorkspacesRequest),
  http.all("*/api/workspaces/parents", unhandledWorkspacesRequest),
  http.all("*/api/auth/workspace-session", unhandledWorkspacesRequest),
];

const installWorkspacesHandlers = async () => {
  vi.resetModules();
  const { WORKSPACES_HANDLERS, resetMockWorkspaces } =
    await import("#/mocks/workspaces-handlers");
  server.resetHandlers(
    ...WORKSPACES_HANDLERS,
    ...UNHANDLED_WORKSPACES_REQUESTS,
  );
  return resetMockWorkspaces;
};

const postJson = (path: string, body: unknown) =>
  fetch(`http://localhost:3000${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

const listWorkspaces = () => fetch("http://localhost:3000/api/workspaces");

const deleteByPath = (path: string, workspacePath: string) =>
  fetch(
    `http://localhost:3000${path}?path=${encodeURIComponent(workspacePath)}`,
    { method: "DELETE" },
  );

describe("mock workspaces handlers", () => {
  afterEach(() => {
    server.resetHandlers();
  });

  it("starts with an empty workspaces list", async () => {
    await installWorkspacesHandlers();
    const response = await listWorkspaces();

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      workspaces: [],
      workspaceParents: [],
    });
  });

  it("persists added workspaces across list calls", async () => {
    await installWorkspacesHandlers();
    await postJson("/api/workspaces", {
      workspaces: [{ id: "w1", name: "Project", path: "/workspace/project" }],
    });

    const response = await listWorkspaces();
    await expect(response.json()).resolves.toEqual({
      workspaces: [{ id: "w1", name: "Project", path: "/workspace/project" }],
      workspaceParents: [],
    });
  });

  it("upserts a workspace when path already exists", async () => {
    await installWorkspacesHandlers();
    await postJson("/api/workspaces", {
      workspaces: [
        { id: "w1", name: "Other", path: "/workspace/other" },
        { id: "w2", name: "Old", path: "/workspace/project" },
      ],
    });
    const response = await postJson("/api/workspaces", {
      workspaces: [
        { id: "w3", name: "New", path: "/workspace/project" },
        { id: "w4", name: "Other updated", path: "/workspace/other" },
      ],
    });

    await expect(response.json()).resolves.toEqual({
      workspaces: [
        { id: "w4", name: "Other updated", path: "/workspace/other" },
        { id: "w3", name: "New", path: "/workspace/project" },
      ],
      workspaceParents: [],
    });
  });

  it("treats omitted workspace and parent collections as no-op updates", async () => {
    await installWorkspacesHandlers();
    const workspacesResponse = await postJson("/api/workspaces", {});

    expect(workspacesResponse.status).toBe(200);
    await expect(workspacesResponse.json()).resolves.toEqual({
      workspaces: [],
      workspaceParents: [],
    });

    const parentsResponse = await postJson("/api/workspaces/parents", {});

    expect(parentsResponse.status).toBe(200);
    await expect(parentsResponse.json()).resolves.toEqual({
      workspaces: [],
      workspaceParents: [],
    });
  });

  it("removes a workspace by path", async () => {
    await installWorkspacesHandlers();
    await postJson("/api/workspaces", {
      workspaces: [
        { id: "w1", name: "Project", path: "/workspace/project" },
        { id: "w2", name: "Other", path: "/workspace/other" },
      ],
    });

    const deleted = await deleteByPath("/api/workspaces", "/workspace/project");

    expect(deleted.status).toBe(200);
    await expect(deleted.json()).resolves.toEqual({ deleted: true });

    const missing = await deleteByPath("/api/workspaces", "/workspace/missing");

    expect(missing.status).toBe(200);
    await expect(missing.json()).resolves.toEqual({ deleted: false });

    const response = await listWorkspaces();
    await expect(response.json()).resolves.toEqual({
      workspaces: [{ id: "w2", name: "Other", path: "/workspace/other" }],
      workspaceParents: [],
    });
  });

  it("persists workspace parents and removes them by path", async () => {
    await installWorkspacesHandlers();
    await postJson("/api/workspaces/parents", {
      parents: [
        { id: "p1", name: "Repos", path: "/workspace/repos" },
        { id: "p2", name: "Other", path: "/workspace/other" },
      ],
    });

    const afterAdd = await listWorkspaces();
    await expect(afterAdd.json()).resolves.toEqual({
      workspaces: [],
      workspaceParents: [
        { id: "p1", name: "Repos", path: "/workspace/repos" },
        { id: "p2", name: "Other", path: "/workspace/other" },
      ],
    });

    const deleted = await deleteByPath(
      "/api/workspaces/parents",
      "/workspace/repos",
    );

    expect(deleted.status).toBe(200);
    await expect(deleted.json()).resolves.toEqual({ deleted: true });

    const missing = await deleteByPath(
      "/api/workspaces/parents",
      "/workspace/missing",
    );

    expect(missing.status).toBe(200);
    await expect(missing.json()).resolves.toEqual({ deleted: false });

    const afterRemove = await listWorkspaces();
    await expect(afterRemove.json()).resolves.toEqual({
      workspaces: [],
      workspaceParents: [{ id: "p2", name: "Other", path: "/workspace/other" }],
    });
  });

  it("replaces a workspace parent when its path is already registered", async () => {
    await installWorkspacesHandlers();
    await postJson("/api/workspaces/parents", {
      parents: [
        { id: "p1", name: "Other", path: "/workspace/other" },
        { id: "p2", name: "Old", path: "/workspace/repos" },
      ],
    });

    const response = await postJson("/api/workspaces/parents", {
      parents: [
        { id: "p3", name: "New", path: "/workspace/repos" },
        { id: "p4", name: "Other updated", path: "/workspace/other" },
      ],
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      workspaces: [],
      workspaceParents: [
        { id: "p4", name: "Other updated", path: "/workspace/other" },
        { id: "p3", name: "New", path: "/workspace/repos" },
      ],
    });
  });

  it("acknowledges workspace session creation and deletion", async () => {
    await installWorkspacesHandlers();
    const createResponse = await fetch(
      "http://localhost:3000/api/auth/workspace-session",
      { method: "POST" },
    );

    expect(createResponse.status).toBe(200);
    await expect(createResponse.json()).resolves.toEqual({ ok: true });

    const deleteResponse = await fetch(
      "http://localhost:3000/api/auth/workspace-session",
      { method: "DELETE" },
    );

    expect(deleteResponse.status).toBe(204);
    await expect(deleteResponse.text()).resolves.toBe("");
  });

  it("resets workspace and parent state together", async () => {
    const resetMockWorkspaces = await installWorkspacesHandlers();
    await postJson("/api/workspaces", {
      workspaces: [{ id: "w1", name: "Project", path: "/workspace/project" }],
    });
    await postJson("/api/workspaces/parents", {
      parents: [{ id: "p1", name: "Repos", path: "/workspace/repos" }],
    });

    resetMockWorkspaces();

    const response = await listWorkspaces();
    await expect(response.json()).resolves.toEqual({
      workspaces: [],
      workspaceParents: [],
    });
  });
});
