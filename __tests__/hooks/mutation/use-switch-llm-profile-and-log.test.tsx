import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

// The wrapper is a thin positional adapter over the base mutation; the inline
// message, metadata persist, and error reporting all live in the mutation
// itself (see use-switch-llm-profile.test.tsx) so they survive the switcher
// menu unmounting on select.
const switchMutateMock = vi.fn();
vi.mock("#/hooks/mutation/use-switch-llm-profile", () => ({
  useSwitchLlmProfile: () => ({ mutate: switchMutateMock, isPending: false }),
}));

import { useSwitchLlmProfileAndLog } from "#/hooks/mutation/use-switch-llm-profile-and-log";

describe("useSwitchLlmProfileAndLog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("maps positional args to the base mutation for the per-conversation path", () => {
    const { result } = renderHook(() => useSwitchLlmProfileAndLog());
    result.current.switchAndLog("conv-1", "claude-sonnet-4.6");

    expect(switchMutateMock).toHaveBeenCalledWith({
      conversationId: "conv-1",
      profileName: "claude-sonnet-4.6",
    });
  });

  it("passes a null conversationId through for the home-page activate path", () => {
    const { result } = renderHook(() => useSwitchLlmProfileAndLog());
    result.current.switchAndLog(null, "claude-sonnet-4.6");

    expect(switchMutateMock).toHaveBeenCalledWith({
      conversationId: null,
      profileName: "claude-sonnet-4.6",
    });
  });
});
