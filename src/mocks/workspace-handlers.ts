import { http, HttpResponse } from "msw";

import { LocalWorkspace, LocalWorkspaceParent } from "#/types/workspace";

interface MockWorkspacesState {
  workspaces: LocalWorkspace[];
  workspaceParents: LocalWorkspaceParent[];
}

const mockWorkspaces: MockWorkspacesState = {
  workspaces: [],
  workspaceParents: [],
};

function workspacesResponse() {
  return {
    workspaces: mockWorkspaces.workspaces,
    workspaceParents: mockWorkspaces.workspaceParents,
  };
}

function readPath(url: URL) {
  return url.searchParams.get("path") ?? "";
}

export function resetWorkspaceMockData() {
  mockWorkspaces.workspaces = [];
  mockWorkspaces.workspaceParents = [];
}

export const WORKSPACE_HANDLERS = [
  http.get("*/api/workspaces", () => HttpResponse.json(workspacesResponse())),
  http.post("*/api/workspaces", async ({ request }) => {
    const body = (await request.json()) as { workspaces?: LocalWorkspace[] };
    const nextWorkspaces = body.workspaces ?? [];
    const existingPaths = new Set(
      mockWorkspaces.workspaces.map((workspace) => workspace.path),
    );

    for (const workspace of nextWorkspaces) {
      if (!existingPaths.has(workspace.path)) {
        mockWorkspaces.workspaces.push(workspace);
        existingPaths.add(workspace.path);
      }
    }

    return HttpResponse.json(workspacesResponse());
  }),
  http.delete("*/api/workspaces", ({ request }) => {
    const path = readPath(new URL(request.url));
    mockWorkspaces.workspaces = mockWorkspaces.workspaces.filter(
      (workspace) => workspace.path !== path,
    );
    return HttpResponse.json({ ok: true });
  }),
  http.post("*/api/workspaces/parents", async ({ request }) => {
    const body = (await request.json()) as { parents?: LocalWorkspaceParent[] };
    const nextParents = body.parents ?? [];
    const existingPaths = new Set(
      mockWorkspaces.workspaceParents.map((parent) => parent.path),
    );

    for (const parent of nextParents) {
      if (!existingPaths.has(parent.path)) {
        mockWorkspaces.workspaceParents.push(parent);
        existingPaths.add(parent.path);
      }
    }

    return HttpResponse.json(workspacesResponse());
  }),
  http.delete("*/api/workspaces/parents", ({ request }) => {
    const path = readPath(new URL(request.url));
    mockWorkspaces.workspaceParents = mockWorkspaces.workspaceParents.filter(
      (parent) => parent.path !== path,
    );
    mockWorkspaces.workspaces = mockWorkspaces.workspaces.filter(
      (workspace) => workspace.parentPath !== path,
    );
    return HttpResponse.json({ ok: true });
  }),
  http.post("*/api/auth/workspace-session", () =>
    HttpResponse.json({ base_url: "/api/conversations/mock/workspace/" }),
  ),
  http.delete(
    "*/api/auth/workspace-session",
    () => new HttpResponse(null, { status: 204 }),
  ),
];
