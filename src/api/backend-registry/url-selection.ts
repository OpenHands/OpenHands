import type { BackendSelection } from "./types";

export const BACKEND_ID_QUERY_PARAM = "backendId";
export const ORG_ID_QUERY_PARAM = "orgId";

export function readBackendSelectionFromSearch(
  search: string,
): BackendSelection | null {
  const params = new URLSearchParams(search);
  const backendId = params.get(BACKEND_ID_QUERY_PARAM)?.trim();
  if (!backendId) return null;

  const orgId = params.get(ORG_ID_QUERY_PARAM)?.trim() || null;
  return { backendId, orgId };
}

export function readBackendSelectionFromLocation(): BackendSelection | null {
  if (typeof window === "undefined") return null;
  return readBackendSelectionFromSearch(window.location.search);
}

export function appendBackendSelectionToUrl(
  path: string,
  selection: BackendSelection | null,
): string {
  if (!selection?.backendId) return path;

  const url = new URL(path, "http://agent-canvas.local");
  url.searchParams.set(BACKEND_ID_QUERY_PARAM, selection.backendId);
  if (selection.orgId) {
    url.searchParams.set(ORG_ID_QUERY_PARAM, selection.orgId);
  } else {
    url.searchParams.delete(ORG_ID_QUERY_PARAM);
  }

  return `${url.pathname}${url.search}${url.hash}`;
}

export function buildConversationUrl(
  conversationId: string,
  selection: BackendSelection | null,
): string {
  return appendBackendSelectionToUrl(
    `/conversations/${conversationId}`,
    selection,
  );
}
