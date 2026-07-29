// Per-(backend, org) memory of the most recently selected conversation.
//
// Used by `BackendSelector` so that flipping from backend A → B while a
// conversation was active in A jumps straight to B's most recent
// conversation (if any) instead of dropping the user on the home page.
// Plain `localStorage` is sufficient: reads happen synchronously during
// the dropdown's onChange and the data is purely a UX shortcut — losing
// it just falls back to the conversations list.

export const LAST_CONVERSATION_STORAGE_KEY =
  "openhands-last-conversation-by-backend";

type StoredMap = Record<string, string>;

function selectionKey(backendId: string, orgId: string | null): string {
  return orgId ? `${backendId}::${orgId}` : backendId;
}

function readMap(): StoredMap {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(LAST_CONVERSATION_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    if (typeof parsed !== "object" || parsed === null) return {};
    const out: StoredMap = {};
    for (const [k, v] of Object.entries(parsed as Record<string, unknown>)) {
      if (typeof v === "string" && v.length > 0) out[k] = v;
    }
    return out;
  } catch {
    return {};
  }
}

function writeMap(map: StoredMap): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(
      LAST_CONVERSATION_STORAGE_KEY,
      JSON.stringify(map),
    );
  } catch {
    /* ignore quota / serialization errors */
  }
}

export function getLastConversationId(
  backendId: string,
  orgId: string | null,
): string | null {
  return readMap()[selectionKey(backendId, orgId)] ?? null;
}

export function setLastConversationId(
  backendId: string,
  orgId: string | null,
  conversationId: string,
): void {
  if (!conversationId) return;
  const map = readMap();
  map[selectionKey(backendId, orgId)] = conversationId;
  writeMap(map);
}

export function clearLastConversationId(
  backendId: string,
  orgId: string | null,
): void {
  const map = readMap();
  const key = selectionKey(backendId, orgId);
  if (key in map) {
    delete map[key];
    writeMap(map);
  }
}
