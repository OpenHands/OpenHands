import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import {
  STAGED_AGENT_NOTIFICATIONS,
  type AgentNotification,
} from "#/components/features/chat/agent-notifications.constants";
import {
  AGENT_NOTIFICATIONS_HISTORY_STORAGE_KEY,
  writeAgentNotificationsHistory,
} from "#/components/features/chat/agent-notifications-storage";
import { useAgentNotifications } from "#/hooks/chat/use-agent-notifications";
import { useAgentNotificationsStore } from "#/stores/use-agent-notifications-store";

const conversationId = "conv-staging-test";

const detectedNotification: AgentNotification = {
  id: "detected-skill-readme",
  kind: "skill",
  name: "README helper",
  prompt: "Save a reusable skill named README helper.",
  createdAt: "2026-01-01T00:00:00.000Z",
};

vi.mock("#/hooks/use-send-message", () => ({
  useSendMessage: () => ({ send: vi.fn() }),
}));

describe("useAgentNotifications staging seed", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_STAGE_AGENT_NOTIFICATIONS", "true");
    window.localStorage.removeItem(AGENT_NOTIFICATIONS_HISTORY_STORAGE_KEY);
    useAgentNotificationsStore.setState({
      historyByConversation: {},
      seenByConversation: {},
    });
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("seeds staged demo rows only before a conversation has persisted history", async () => {
    renderHook(() =>
      useAgentNotifications({ conversationId, enabled: true }),
    );

    await waitFor(() => {
      expect(
        useAgentNotificationsStore.getState().historyByConversation[
          conversationId
        ],
      ).toEqual(STAGED_AGENT_NOTIFICATIONS);
    });
  });

  it("does not re-seed staged demo rows after the user deletes the last item", async () => {
    writeAgentNotificationsHistory(conversationId, [detectedNotification]);
    useAgentNotificationsStore.setState({
      historyByConversation: { [conversationId]: [detectedNotification] },
      seenByConversation: { [conversationId]: [] },
    });

    const { result } = renderHook(() =>
      useAgentNotifications({ conversationId, enabled: true }),
    );

    result.current.remove(detectedNotification.id);

    await waitFor(() => {
      expect(
        useAgentNotificationsStore.getState().historyByConversation[
          conversationId
        ],
      ).toEqual([]);
    });

    expect(
      useAgentNotificationsStore.getState().historyByConversation[conversationId],
    ).toEqual([]);
  });
});
