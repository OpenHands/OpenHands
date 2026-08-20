import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import { useAgentNotification } from "#/hooks/use-agent-notification";
import { AgentState } from "#/types/agent-state";

// Mock useSettings to control the sound notification setting
vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: vi.fn().mockReturnValue({
    data: { enable_sound_notifications: true },
  }),
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
