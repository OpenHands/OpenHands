import type { AgentNotification } from "./agent-notifications.constants";

export const AGENT_NOTIFICATIONS_HISTORY_STORAGE_KEY =
  "openhands-agent-notifications-history";

export const AGENT_NOTIFICATIONS_SEEN_STORAGE_KEY =
  "openhands-agent-notifications-seen";

function readJsonMap<T>(key: string): Record<string, T> {
  if (typeof window === "undefined") {
    return {};
  }

  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) {
      return {};
    }

    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) {
      return {};
    }

    return parsed as Record<string, T>;
  } catch {
    return {};
  }
}

function writeJsonMap<T>(key: string, map: Record<string, T>): void {
  window.localStorage.setItem(key, JSON.stringify(map));
}

/** All agent notifications ever detected for this conversation, oldest first. */
export function readAgentNotificationsHistory(
  conversationId: string,
): AgentNotification[] {
  if (!conversationId) {
    return [];
  }

  return (
    readJsonMap<AgentNotification[]>(AGENT_NOTIFICATIONS_HISTORY_STORAGE_KEY)[
      conversationId
    ] ?? []
  );
}

/** True when this conversation has ever been written to history storage. */
export function hasAgentNotificationsHistoryEntry(
  conversationId: string,
): boolean {
  if (!conversationId) {
    return false;
  }

  const map = readJsonMap<AgentNotification[]>(
    AGENT_NOTIFICATIONS_HISTORY_STORAGE_KEY,
  );
  return Object.prototype.hasOwnProperty.call(map, conversationId);
}

export function writeAgentNotificationsHistory(
  conversationId: string,
  notifications: AgentNotification[],
): void {
  if (!conversationId) {
    return;
  }

  const map = readJsonMap<AgentNotification[]>(
    AGENT_NOTIFICATIONS_HISTORY_STORAGE_KEY,
  );
  map[conversationId] = notifications;
  writeJsonMap(AGENT_NOTIFICATIONS_HISTORY_STORAGE_KEY, map);
}

/** Ids of notifications already surfaced above the chat input at least once. */
export function readSeenAgentNotificationIds(conversationId: string): string[] {
  if (!conversationId) {
    return [];
  }

  return (
    readJsonMap<string[]>(AGENT_NOTIFICATIONS_SEEN_STORAGE_KEY)[
      conversationId
    ] ?? []
  );
}

export function writeSeenAgentNotificationIds(
  conversationId: string,
  ids: string[],
): void {
  if (!conversationId) {
    return;
  }

  const map = readJsonMap<string[]>(AGENT_NOTIFICATIONS_SEEN_STORAGE_KEY);
  map[conversationId] = ids;
  writeJsonMap(AGENT_NOTIFICATIONS_SEEN_STORAGE_KEY, map);
}
