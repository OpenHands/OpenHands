import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import { useAgentNotification } from "#/hooks/use-agent-notification";
import { useSettings } from "#/hooks/query/use-settings";
import { AgentState } from "#/types/agent-state";
import { I18nKey } from "#/i18n/declaration";

const mockNavigate = vi.hoisted(() => vi.fn());
const mockSetActive = vi.hoisted(() => vi.fn());
const mockActive = vi.hoisted(() => ({
  backendId: "backend-a",
  orgId: "org-a",
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: vi.fn(),
}));

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({
    conversationId: "conversation-123",
    navigate: mockNavigate,
  }),
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackendContext: () => ({
    active: { backend: { id: mockActive.backendId }, orgId: mockActive.orgId },
    setActive: mockSetActive,
  }),
}));

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

class MockNotification {
  static permission: NotificationPermission = "granted";

  static requestPermission = vi.fn<() => Promise<NotificationPermission>>();

  static instances: MockNotification[] = [];

  onclick: (() => void) | null = null;

  close = vi.fn();

  constructor(
    readonly title: string,
    readonly options?: NotificationOptions,
  ) {
    MockNotification.instances.push(this);
  }
}

vi.stubGlobal("Audio", MockAudio);
vi.stubGlobal("Notification", MockNotification);

type HookProps = {
  state: AgentState;
  conversationTitle: string;
};

const conversationTitle = "Fix the tests";
let visibilityState: DocumentVisibilityState;

const renderNotificationHook = () =>
  renderHook(
    ({ state, conversationTitle: currentConversationTitle }: HookProps) =>
      useAgentNotification(state, currentConversationTitle),
    {
      initialProps: { state: AgentState.RUNNING, conversationTitle },
    },
  );

describe("useAgentNotification", () => {
  const mockUseSettings = vi.mocked(useSettings);

  beforeEach(() => {
    vi.clearAllMocks();
    MockNotification.instances.length = 0;
    MockNotification.permission = "granted";
    mockActive.backendId = "backend-a";
    mockActive.orgId = "org-a";
    mockUseSettings.mockReturnValue({
      data: {
        enable_sound_notifications: true,
        enable_desktop_notifications: true,
      },
    } as ReturnType<typeof useSettings>);
    visibilityState = "hidden";
    vi.spyOn(document, "visibilityState", "get").mockImplementation(
      () => visibilityState,
    );
    vi.spyOn(window, "focus").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it.each([
    AgentState.FINISHED,
    AgentState.AWAITING_USER_INPUT,
    AgentState.AWAITING_USER_CONFIRMATION,
  ])("plays a sound on a transition into %s", (state) => {
    const { rerender } = renderNotificationHook();

    rerender({ state, conversationTitle });

    expect(mockPlay).toHaveBeenCalledTimes(1);
  });

  it("does not play sound when sound notifications are disabled", () => {
    mockUseSettings.mockReturnValue({
      data: {
        enable_sound_notifications: false,
        enable_desktop_notifications: false,
      },
    } as ReturnType<typeof useSettings>);
    const { rerender } = renderNotificationHook();

    rerender({ state: AgentState.FINISHED, conversationTitle });

    expect(mockPlay).not.toHaveBeenCalled();
  });

  it.each([
    [AgentState.FINISHED, I18nKey.DESKTOP_NOTIFICATIONS$FINISHED],
    [
      AgentState.AWAITING_USER_INPUT,
      I18nKey.DESKTOP_NOTIFICATIONS$AWAITING_USER_INPUT,
    ],
    [
      AgentState.AWAITING_USER_CONFIRMATION,
      I18nKey.DESKTOP_NOTIFICATIONS$AWAITING_USER_CONFIRMATION,
    ],
  ] as const)(
    "shows one desktop notification per transition into %s while the tab is hidden",
    (state, body) => {
      const { rerender } = renderNotificationHook();

      rerender({ state, conversationTitle });

      expect(MockNotification.instances).toHaveLength(1);
      expect(MockNotification.instances[0]).toMatchObject({
        title: "Fix the tests",
        options: { body },
      });

      rerender({
        state,
        conversationTitle: "Updated title",
      });

      expect(MockNotification.instances).toHaveLength(1);
    },
  );

  it("does not show a desktop notification while the tab is visible", () => {
    visibilityState = "visible";
    const { rerender } = renderNotificationHook();

    rerender({ state: AgentState.AWAITING_USER_INPUT, conversationTitle });

    expect(MockNotification.instances).toHaveLength(0);
  });

  it.each(["default", "denied"] as NotificationPermission[])(
    "does not show a desktop notification when permission is %s",
    (permission) => {
      MockNotification.permission = permission;
      const { rerender } = renderNotificationHook();

      rerender({ state: AgentState.FINISHED, conversationTitle });

      expect(MockNotification.instances).toHaveLength(0);
    },
  );

  it("does not show a desktop notification when the setting is disabled", () => {
    mockUseSettings.mockReturnValue({
      data: {
        enable_sound_notifications: false,
        enable_desktop_notifications: false,
      },
    } as ReturnType<typeof useSettings>);
    const { rerender } = renderNotificationHook();

    rerender({ state: AgentState.FINISHED, conversationTitle });

    expect(MockNotification.instances).toHaveLength(0);
  });

  it("restores the originating backend before opening the conversation", () => {
    const { rerender } = renderNotificationHook();
    rerender({
      state: AgentState.AWAITING_USER_CONFIRMATION,
      conversationTitle,
    });
    mockActive.backendId = "backend-b";
    mockActive.orgId = "org-b";
    rerender({
      state: AgentState.AWAITING_USER_CONFIRMATION,
      conversationTitle,
    });

    MockNotification.instances[0]?.onclick?.();

    expect(window.focus).toHaveBeenCalledTimes(1);
    expect(mockSetActive).toHaveBeenCalledWith("backend-a", "org-a");
    expect(mockNavigate).toHaveBeenCalledWith(
      "/conversations/conversation-123",
    );
    expect(mockSetActive.mock.invocationCallOrder[0]).toBeLessThan(
      mockNavigate.mock.invocationCallOrder[0],
    );
    expect(MockNotification.instances[0]?.close).toHaveBeenCalledTimes(1);
  });

  it("does not trigger for non-notification states", () => {
    const { rerender } = renderNotificationHook();

    rerender({ state: AgentState.LOADING, conversationTitle });

    expect(mockPlay).not.toHaveBeenCalled();
    expect(MockNotification.instances).toHaveLength(0);
  });
});
