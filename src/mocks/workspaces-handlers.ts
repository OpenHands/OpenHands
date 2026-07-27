import { http, HttpResponse } from "msw";
import type {
  WorkspaceItem,
  WorkspaceParentItem,
  WorkspacesListResponse,
} from "@openhands/typescript-client/clients";

/**
 * MSW handlers for the agent-server's `/api/workspaces` and
 * `/api/auth/workspace-session` endpoints used by `useLocalWorkspaces`,
 * `WorkspacesService`, and `useWorkspaceSession`.
 *
 * Without these handlers, mock-mode renders of the home / sidebar /
 * conversation pages let those requests fall through to the Vite proxy
 * (`127.0.0.1:8000` ECONNREFUSED), which spams logs and surfaces
 * global error toasts in mock-mode screens.
 *
 * The mock keeps an in-memory list so the install / remove flow round-trips
 * cleanly within a single page load. State resets on every test reload.
 */

let workspaces: WorkspaceItem[] = [];
let workspaceParents: WorkspaceParentItem[] = [];

function snapshot(): WorkspacesListResponse {
  return {
    workspaces: workspaces.map((w) => ({ ...w })),
    workspaceParents: workspaceParents.map((p) => ({ ...p })),
  };
}

export function resetMockWorkspaces(): void {
  workspaces = [];
  workspaceParents = [];
}

export const WORKSPACES_HANDLERS = [
  http.get("*/api/workspaces", async () => HttpResponse.json(snapshot())),

  http.post("*/api/workspaces", async ({ request }) => {
    // `workspaces` is the top-level key used by WorkspacesClient.addWorkspaces()
    // in @openhands/typescript-client — aligns with the agent-server SDK contract.
    const body = (await request.json()) as { workspaces?: WorkspaceItem[] };
    for (const incoming of body.workspaces ?? []) {
      const existingIndex = workspaces.findIndex(
        (w) => w.path === incoming.path,
      );
      if (existingIndex >= 0) {
        workspaces[existingIndex] = { ...incoming };
      } else {
        workspaces.push({ ...incoming });
      }
    }
    return HttpResponse.json(snapshot());
  }),

  http.delete("*/api/workspaces", async ({ request }) => {
    const url = new URL(request.url);
    const path = url.searchParams.get("path");
    const before = workspaces.length;
    workspaces = workspaces.filter((w) => w.path !== path);
    // WorkspacesService.removeWorkspace() discards the response body (returns
    // void), so the shape here is mock-only and not coupled to the real contract.
    return HttpResponse.json({ deleted: workspaces.length !== before });
  }),

  http.post("*/api/workspaces/parents", async ({ request }) => {
    const body = (await request.json()) as { parents?: WorkspaceParentItem[] };
    for (const incoming of body.parents ?? []) {
      const existingIndex = workspaceParents.findIndex(
        (p) => p.path === incoming.path,
      );
      if (existingIndex >= 0) {
        workspaceParents[existingIndex] = { ...incoming };
      } else {
        workspaceParents.push({ ...incoming });
      }
    }
    return HttpResponse.json(snapshot());
  }),

  http.delete("*/api/workspaces/parents", async ({ request }) => {
    const url = new URL(request.url);
    const path = url.searchParams.get("path");
    const before = workspaceParents.length;
    workspaceParents = workspaceParents.filter((p) => p.path !== path);
    // WorkspacesService.removeWorkspaceParent() discards the response body
    // (returns void), so the shape here is mock-only and not coupled to the
    // real contract.
    return HttpResponse.json({ deleted: workspaceParents.length !== before });
  }),

  // Workspace static-asset session: minted on conversation pages by
  // `useWorkspaceSession`. The real endpoint sets an HttpOnly cookie; in
  // mock mode we just acknowledge the POST/DELETE so the call doesn't fall
  // through to the Vite proxy.
  http.post("*/api/auth/workspace-session", async () =>
    HttpResponse.json({ ok: true }),
  ),

  http.delete(
    "*/api/auth/workspace-session",
    async () => new HttpResponse(null, { status: 204 }),
  ),
];
