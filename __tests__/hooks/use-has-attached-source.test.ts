import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { renderHook } from "@testing-library/react";

import { useHasAttachedSource } from "#/hooks/use-has-attached-source";
import { setStoredConversationMetadata } from "#/api/conversation-metadata-store";

const useActiveConversationMock = vi.fn();

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => useActiveConversationMock(),
}));

describe("useHasAttachedSource", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useActiveConversationMock.mockReset();
  });

  afterEach(() => {
    window.localStorage.clear();
  });

  it("returns true when the conversation has a selected_repository", () => {
    useActiveConversationMock.mockReturnValue({
      data: {
        id: "conv-1",
        selected_repository: "octocat/hello-world",
      },
      isLoading: false,
    });

    const { result } = renderHook(() => useHasAttachedSource());

    expect(result.current.hasAttachedSource).toBe(true);
    expect(result.current.isLoading).toBe(false);
  });

  it("returns true when the conversation has a stored selected_workspace (no repo)", () => {
    setStoredConversationMetadata("conv-2", {
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
      selected_workspace: "/home/me/code/foo",
    });

    useActiveConversationMock.mockReturnValue({
      data: { id: "conv-2", selected_repository: null },
      isLoading: false,
    });

    const { result } = renderHook(() => useHasAttachedSource());

    expect(result.current.hasAttachedSource).toBe(true);
  });

  it("returns false when neither a repo nor a workspace is attached", () => {
    useActiveConversationMock.mockReturnValue({
      data: { id: "conv-3", selected_repository: null },
      isLoading: false,
    });

    const { result } = renderHook(() => useHasAttachedSource());

    expect(result.current.hasAttachedSource).toBe(false);
  });

  it("propagates the active-conversation isLoading flag", () => {
    useActiveConversationMock.mockReturnValue({
      data: undefined,
      isLoading: true,
    });

    const { result } = renderHook(() => useHasAttachedSource());

    expect(result.current.hasAttachedSource).toBe(false);
    expect(result.current.isLoading).toBe(true);
  });
});
