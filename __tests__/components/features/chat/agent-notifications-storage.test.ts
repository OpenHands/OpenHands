import { beforeEach, describe, expect, it } from "vitest";
import type { AgentNotification } from "#/components/features/chat/agent-notifications.constants";
import {
  AGENT_NOTIFICATIONS_HISTORY_STORAGE_KEY,
  AGENT_NOTIFICATIONS_SEEN_STORAGE_KEY,
  readAgentNotificationsHistory,
  readSeenAgentNotificationIds,
  writeAgentNotificationsHistory,
  writeSeenAgentNotificationIds,
} from "#/components/features/chat/agent-notifications-storage";

const sampleNotification: AgentNotification = {
  id: "skill-standup",
  kind: "skill",
  name: "Standup digest helper",
  prompt: "Save a reusable skill.",
  createdAt: "2026-01-01T00:00:00.000Z",
};

describe("agent-notifications-storage", () => {
  beforeEach(() => {
    window.localStorage.removeItem(AGENT_NOTIFICATIONS_HISTORY_STORAGE_KEY);
    window.localStorage.removeItem(AGENT_NOTIFICATIONS_SEEN_STORAGE_KEY);
  });

  it("reads and writes notification history per conversation id", () => {
    expect(readAgentNotificationsHistory("conv-a")).toEqual([]);

    writeAgentNotificationsHistory("conv-a", [sampleNotification]);
    writeAgentNotificationsHistory("conv-b", [sampleNotification]);

    expect(readAgentNotificationsHistory("conv-a")).toEqual([sampleNotification]);
    expect(readAgentNotificationsHistory("conv-b")).toEqual([sampleNotification]);
    expect(readAgentNotificationsHistory("conv-c")).toEqual([]);
  });

  it("reads and writes seen notification ids per conversation id", () => {
    expect(readSeenAgentNotificationIds("conv-a")).toEqual([]);

    writeSeenAgentNotificationIds("conv-a", ["skill-standup"]);
    writeSeenAgentNotificationIds("conv-b", ["workflow-ci"]);

    expect(readSeenAgentNotificationIds("conv-a")).toEqual(["skill-standup"]);
    expect(readSeenAgentNotificationIds("conv-b")).toEqual(["workflow-ci"]);
    expect(readSeenAgentNotificationIds("conv-c")).toEqual([]);
  });
});
