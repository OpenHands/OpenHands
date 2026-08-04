export const AGENT_NOTIFICATIONS_PENDING_PROMPTS_STORAGE_KEY =
  "openhands-agent-notifications-pending-prompts";

type PendingPromptsMap = Record<string, string[]>;

function readPendingPromptsMap(): PendingPromptsMap {
  if (typeof window === "undefined") {
    return {};
  }

  try {
    const raw = window.sessionStorage.getItem(
      AGENT_NOTIFICATIONS_PENDING_PROMPTS_STORAGE_KEY,
    );
    if (!raw) {
      return {};
    }

    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) {
      return {};
    }

    return Object.fromEntries(
      Object.entries(parsed).filter(
        (entry): entry is [string, string[]] =>
          Array.isArray(entry[1]) &&
          entry[1].every((prompt) => typeof prompt === "string"),
      ),
    );
  } catch {
    return {};
  }
}

function writePendingPromptsMap(map: PendingPromptsMap): void {
  window.sessionStorage.setItem(
    AGENT_NOTIFICATIONS_PENDING_PROMPTS_STORAGE_KEY,
    JSON.stringify(map),
  );
}

export function readAgentNotificationPendingPrompts(
  conversationId: string,
): string[] {
  if (!conversationId) {
    return [];
  }

  return readPendingPromptsMap()[conversationId] ?? [];
}

export function writeAgentNotificationPendingPrompts(
  conversationId: string,
  prompts: string[],
): void {
  if (!conversationId || prompts.length === 0) {
    return;
  }

  const map = readPendingPromptsMap();
  map[conversationId] = prompts;
  writePendingPromptsMap(map);
}

export function clearAgentNotificationPendingPrompts(
  conversationId: string,
): void {
  if (!conversationId) {
    return;
  }

  const map = readPendingPromptsMap();
  delete map[conversationId];
  writePendingPromptsMap(map);
}
