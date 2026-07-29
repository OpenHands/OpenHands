import { QueryClient } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ActionEvent } from "#/types/agent-server/core/events/action-event";
import { handleActionEventCacheInvalidation } from "#/utils/cache-utils";
import { useModelStore } from "#/stores/model-store";

const makeActionEvent = (overrides: Partial<ActionEvent>): ActionEvent =>
  ({
    id: "ev-1",
    timestamp: new Date().toISOString(),
    source: "agent",
    tool_name: "SwitchLLMTool",
    tool_call_id: "call-1",
    action: { kind: "SwitchLLMAction" },
    ...overrides,
  }) as unknown as ActionEvent;

describe("handleActionEventCacheInvalidation", () => {
  beforeEach(() => {
    useModelStore.setState({
      entriesByConversation: {},
      activeProfileByConversation: {},
    });
  });

  it("refreshes the conversation and drops the optimistic profile when SwitchLLMTool fires", () => {
    useModelStore.setState({
      activeProfileByConversation: { "conv-1": "haiku" },
    });
    const queryClient = new QueryClient();
    const spy = vi.spyOn(queryClient, "invalidateQueries");

    handleActionEventCacheInvalidation(
      makeActionEvent({ tool_name: "SwitchLLMTool" }),
      "conv-1",
      queryClient,
    );

    expect(spy).toHaveBeenCalledWith({
      queryKey: ["user", "conversation", "conv-1"],
    });
    expect(
      useModelStore.getState().activeProfileByConversation["conv-1"],
    ).toBeUndefined();
  });

  it("does not touch the conversation cache for unrelated tool events", () => {
    const queryClient = new QueryClient();
    const spy = vi.spyOn(queryClient, "invalidateQueries");

    handleActionEventCacheInvalidation(
      makeActionEvent({ tool_name: "terminal" }),
      "conv-1",
      queryClient,
    );

    const conversationInvalidations = spy.mock.calls.filter(
      ([arg]) =>
        Array.isArray((arg as { queryKey?: unknown[] })?.queryKey) &&
        (arg as { queryKey: unknown[] }).queryKey[0] === "user",
    );
    expect(conversationInvalidations).toHaveLength(0);
  });
});
