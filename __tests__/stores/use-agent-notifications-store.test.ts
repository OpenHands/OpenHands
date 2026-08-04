import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { STAGED_AGENT_NOTIFICATIONS } from "#/components/features/chat/agent-notifications.constants";
import {
  AGENT_NOTIFICATIONS_HISTORY_STORAGE_KEY,
  writeAgentNotificationsHistory,
} from "#/components/features/chat/agent-notifications-storage";
import { useAgentNotificationsStore } from "#/stores/use-agent-notifications-store";

const conversationId = "conv-live";

describe("useAgentNotificationsStore staged purge", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_STAGE_AGENT_NOTIFICATIONS", "");
    window.localStorage.removeItem(AGENT_NOTIFICATIONS_HISTORY_STORAGE_KEY);
    useAgentNotificationsStore.setState({
      historyByConversation: {},
      seenByConversation: {},
    });
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("drops persisted staged demo rows when staging is disabled", () => {
    writeAgentNotificationsHistory(conversationId, STAGED_AGENT_NOTIFICATIONS);

    useAgentNotificationsStore.getState().ensureHydrated(conversationId);

    expect(
      useAgentNotificationsStore.getState().historyByConversation[conversationId],
    ).toEqual([]);
    expect(readAgentNotificationsHistory(conversationId)).toEqual([]);
  });
});

function readAgentNotificationsHistory(conversationId: string) {
  const raw = window.localStorage.getItem(AGENT_NOTIFICATIONS_HISTORY_STORAGE_KEY);
  if (!raw) {
    return [];
  }
  const parsed = JSON.parse(raw) as Record<string, unknown>;
  return (parsed[conversationId] as unknown[]) ?? [];
}
