import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import { useAgentNotification } from "#/hooks/use-agent-notification";
import { AgentState } from "#/types/agent-state";

// Mock useSettings to control the notification settings
vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: vi.fn().mockReturnValue({
    data: {
      enable_sound_notifications: true,
      enable_desktop_notifications: false,
    },
  }),
}));

// Mock the conversation-title helper default so placeholder detection is
// deterministic without pulling the agent-server adapter in.
vi.mock("#/utils/conversation-title", () => ({
  isPlaceholderConversationTitle: (title: string | null | undefined) =>
    !title?.trim() || title.trim() === "Conversation",
}));

// Mock Audio
const mockPlay = vi.fn().mockResolvedValue(undefined);
const mockAudio = {
  play: mockPlay,
  currentTime: 0,
  volume: 0.5,
};

class MockAudio {
  play = mockPlay;
  currentTime = 0;
  volume = 0.5;
  constructor() {
    Object.assign(this, mockAudio);
    return mockAudio as unknown as MockAudio;
  }
}
vi.stubGlobal("Audio", MockAudio);

describe("useAgentNotification", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("plays notification sound when agent reaches FINISHED state and sound is enabled", () => {
    const { rerender } = renderHook(
      ({ state }) => useAgentNotification(state),
      { initialProps: { state: AgentState.RUNNING } },
    );

    rerender({ state: AgentState.FINISHED });

    expect(mockPlay).toHaveBeenCalledTimes(1);
  });

  it("plays notification sound when agent reaches AWAITING_USER_INPUT state", () => {
    const { rerender } = renderHook(
      ({ state }) => useAgentNotification(state),
      { initialProps: { state: AgentState.RUNNING } },
    );

    rerender({ state: AgentState.AWAITING_USER_INPUT });

    expect(mockPlay).toHaveBeenCalledTimes(1);
  });

  it("plays notification sound when agent reaches AWAITING_USER_CONFIRMATION state", () => {
    const { rerender } = renderHook(
      ({ state }) => useAgentNotification(state),
      { initialProps: { state: AgentState.RUNNING } },
    );

    rerender({ state: AgentState.AWAITING_USER_CONFIRMATION });

    expect(mockPlay).toHaveBeenCalledTimes(1);
  });

  it("does not play sound when sound notifications are disabled", async () => {
    const { useSettings } = await import("#/hooks/query/use-settings");
    vi.mocked(useSettings).mockReturnValue({
      data: { enable_sound_notifications: false },
    } as ReturnType<typeof useSettings>);

    const { rerender } = renderHook(
      ({ state }) => useAgentNotification(state),
      { initialProps: { state: AgentState.RUNNING } },
    );

    rerender({ state: AgentState.FINISHED });

    expect(mockPlay).not.toHaveBeenCalled();

    // Restore
    vi.mocked(useSettings).mockReturnValue({
      data: { enable_sound_notifications: true },
    } as ReturnType<typeof useSettings>);
  });

  it("does not trigger for non-notification states like RUNNING", () => {
    const { rerender } = renderHook(
      ({ state }) => useAgentNotification(state),
      { initialProps: { state: AgentState.LOADING } },
    );

    rerender({ state: AgentState.RUNNING });

    expect(mockPlay).not.toHaveBeenCalled();
  });
});

class MockNotification {
  static permission: NotificationPermission = "granted";
  static instances: MockNotification[] = [];

  onclick: (() => void) | null = null;
  close = vi.fn();

  constructor(
    public title: string,
    public options?: NotificationOptions,
  ) {
    MockNotification.instances.push(this);
  }
}

function stubNotification(permission: NotificationPermission) {
  MockNotification.permission = permission;
  MockNotification.instances = [];
  vi.stubGlobal("Notification", MockNotification);
}

function setTabHidden(hidden: boolean) {
  Object.defineProperty(document, "hidden", {
    configurable: true,
    value: hidden,
  });
}

async function setSettings(overrides: Record<string, unknown>) {
  const { useSettings } = await import("#/hooks/query/use-settings");
  vi.mocked(useSettings).mockReturnValue({
    data: {
      enable_sound_notifications: false,
      enable_desktop_notifications: false,
      ...overrides,
    },
  } as ReturnType<typeof useSettings>);
}

describe("useAgentNotification desktop notifications", () => {
  const focusSpy = vi.spyOn(window, "focus").mockImplementation(() => {});

  beforeEach(() => {
    vi.clearAllMocks();
    focusSpy.mockClear();
    setTabHidden(true);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    Object.defineProperty(document, "hidden", { configurable: true, value: false });
  });

  it("shows one desktop notification for a transition in a backgrounded tab, naming the conversation, and focuses the window on click", async () => {
    await setSettings({ enable_desktop_notifications: true });
    stubNotification("granted");

    const { rerender } = renderHook(
      ({ state, context }) => useAgentNotification(state, context),
      {
        initialProps: {
          state: AgentState.RUNNING,
          context: {
            conversationId: "conv-1",
            conversationTitle: "Fix the login bug",
          },
        },
      },
    );
    rerender({
      state: AgentState.FINISHED,
      context: {
        conversationId: "conv-1",
        conversationTitle: "Fix the login bug",
      },
    });

    expect(MockNotification.instances).toHaveLength(1);
    const notification = MockNotification.instances[0];
    expect(notification.title).toBe("Fix the login bug");
    expect(notification.options?.body).toBe("NOTIFICATION$FINISHED");
    expect(notification.options?.tag).toBe("conv-1");

    notification.onclick?.();
    expect(focusSpy).toHaveBeenCalledTimes(1);
    expect(notification.close).toHaveBeenCalledTimes(1);
  });

  it("does not show a desktop notification while the tab is visible", async () => {
    await setSettings({ enable_desktop_notifications: true });
    stubNotification("granted");
    setTabHidden(false);

    const { rerender } = renderHook(
      ({ state }) => useAgentNotification(state),
      { initialProps: { state: AgentState.RUNNING } },
    );
    rerender({ state: AgentState.AWAITING_USER_INPUT });

    expect(MockNotification.instances).toHaveLength(0);
  });

  it("does not show a desktop notification when the setting is disabled", async () => {
    await setSettings({ enable_desktop_notifications: false });
    stubNotification("granted");

    const { rerender } = renderHook(
      ({ state }) => useAgentNotification(state),
      { initialProps: { state: AgentState.RUNNING } },
    );
    rerender({ state: AgentState.FINISHED });

    expect(MockNotification.instances).toHaveLength(0);
  });

  it("does not show a desktop notification for non-notification states or repeated renders of an unchanged state", async () => {
    await setSettings({ enable_desktop_notifications: true });
    stubNotification("granted");

    const { rerender } = renderHook(
      ({ state }) => useAgentNotification(state),
      { initialProps: { state: AgentState.LOADING } },
    );
    rerender({ state: AgentState.RUNNING });
    rerender({ state: AgentState.RUNNING });
    rerender({ state: AgentState.AWAITING_USER_INPUT });
    rerender({ state: AgentState.AWAITING_USER_INPUT });

    expect(MockNotification.instances).toHaveLength(1);
    expect(MockNotification.instances[0].options?.body).toBe(
      "NOTIFICATION$AWAITING_INPUT",
    );
  });

  it("uses a distinct body per attention state and falls back to an untitled label without a conversation title", async () => {
    await setSettings({ enable_desktop_notifications: true });
    stubNotification("granted");

    const { rerender } = renderHook(
      ({ state, context }) => useAgentNotification(state, context),
      {
        initialProps: {
          state: AgentState.RUNNING,
          context: { conversationId: "conv-2", conversationTitle: null },
        },
      },
    );

    rerender({
      state: AgentState.AWAITING_USER_INPUT,
      context: { conversationId: "conv-2", conversationTitle: null },
    });
    rerender({
      state: AgentState.RUNNING,
      context: { conversationId: "conv-2", conversationTitle: null },
    });
    rerender({
      state: AgentState.AWAITING_USER_CONFIRMATION,
      context: { conversationId: "conv-2", conversationTitle: null },
    });

    expect(MockNotification.instances).toHaveLength(2);
    expect(MockNotification.instances[0].options?.body).toBe(
      "NOTIFICATION$AWAITING_INPUT",
    );
    expect(MockNotification.instances[1].options?.body).toBe(
      "NOTIFICATION$CONFIRMATION",
    );
    expect(MockNotification.instances[0].title).toBe(
      "NOTIFICATION$UNTITLED_CONVERSATION",
    );
  });

  it("does not show a desktop notification without granted permission", async () => {
    await setSettings({ enable_desktop_notifications: true });
    stubNotification("denied");

    const { rerender } = renderHook(
      ({ state }) => useAgentNotification(state),
      { initialProps: { state: AgentState.RUNNING } },
    );
    rerender({ state: AgentState.FINISHED });

    expect(MockNotification.instances).toHaveLength(0);
  });
});
