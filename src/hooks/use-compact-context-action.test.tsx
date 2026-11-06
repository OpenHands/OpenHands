/* Integration: useCompactContextAction end-to-end through real stores. */
import { act, renderHook } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import useMetricsStore from "#/stores/metrics-store";
import { useEventStore } from "#/stores/use-event-store";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import type { Backend } from "#/api/backend-registry/types";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { AgentState } from "#/types/agent-state";
import { subscribeConversationContextChangeRequested } from "#/services/conversation-context-events";

import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { createAgentServerQueryClient } from "#/query-client-config";
import { HttpError } from "@openhands/typescript-client";
const activeConversation = {
  id: "c1",
  conversation_url: null,
  session_api_key: null,
  agent_kind: "openhands" as const,
};
let conversation: Pick<
  AppConversation,
  "id" | "conversation_url" | "session_api_key" | "agent_kind"
> | null;
let agentState: AgentState;
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({ data: conversation }),
}));
vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: () => ({ curAgentState: agentState }),
}));
const successToast = vi.fn();
const errorToast = vi.fn();
vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: (...args: unknown[]) => successToast(...args),
  displayErrorToast: (...args: unknown[]) => errorToast(...args),
}));

import { useCompactContextAction } from "#/hooks/use-compact-context-action";
import { I18nKey } from "#/i18n/declaration";

function condensationEvent(id: string): OpenHandsEvent {
  return {
    id,
    timestamp: new Date().toISOString(),
    source: "environment",
    kind: "Condensation",
    forgotten_event_ids: ["a"],
  } as OpenHandsEvent;
}

const usage = (perTurn: number) => ({
  prompt_tokens: perTurn,
  completion_tokens: 0,
  cache_read_tokens: 0,
  cache_write_tokens: 0,
  context_window: 262144,
  per_turn_token: perTurn,
});

const backend: Backend = {
  id: "compaction-owner",
  name: "Compaction test",
  host: "http://agent-server.test",
  apiKey: "",
  kind: "local",
  connectionRevision: 7,
};
const scope = {
  backendId: backend.id,
  orgId: null,
  connectionRevision: 7,
};
const unsubscribe: Array<() => void> = [];

function wrapper({ children }: { children: React.ReactNode }) {
  return React.createElement(
    QueryClientProvider,
    { client: createAgentServerQueryClient() },
    React.createElement(ActiveBackendProvider, null, children),
  );
}

describe("useCompactContextAction (integration)", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    conversation = activeConversation;
    agentState = AgentState.FINISHED;
    setRegisteredBackends([backend]);
    setActiveSelection({ backendId: backend.id });
    useEventStore.getState().clearEvents();
    useMetricsStore.getState().setMetrics({
      cost: null,
      max_budget_per_task: null,
      usage: usage(50_000),
    });
    vi.spyOn(
      AgentServerConversationService,
      "condenseConversation",
    ).mockResolvedValue(undefined);
    successToast.mockClear();
    errorToast.mockClear();
  });

  afterEach(() => {
    unsubscribe.splice(0).forEach((dispose) => dispose());
    vi.restoreAllMocks();
    vi.useRealTimers();
    useEventStore.getState().clearEvents();
    setActiveSelection(null);
    setRegisteredBackends([]);
  });

  it("signals only the owning backend/org/revision synchronously before requesting compaction", async () => {
    const order: string[] = [];
    const onRequested = vi.fn(() => order.push("context-change"));
    const foreignListener = vi.fn();
    unsubscribe.push(
      subscribeConversationContextChangeRequested(scope, onRequested),
      ...[
        { ...scope, backendId: "other-backend" },
        { ...scope, orgId: "other-org" },
        { ...scope, connectionRevision: 8 },
      ].map((otherScope) =>
        subscribeConversationContextChangeRequested(
          otherScope,
          foreignListener,
        ),
      ),
    );
    vi.mocked(
      AgentServerConversationService.condenseConversation,
    ).mockImplementation(async () => {
      order.push("condense-request");
    });
    const { result } = renderHook(() => useCompactContextAction(), { wrapper });

    await act(async () => {
      result.current.handleCompact();
      expect(order).toEqual(["context-change"]);
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(onRequested).toHaveBeenCalledExactlyOnceWith({
      conversationId: "c1",
      reason: "condense",
    });
    expect(order).toEqual(["context-change", "condense-request"]);
    expect(foreignListener).not.toHaveBeenCalled();
  });

  it.each([
    "no-conversation",
    "acp",
    "running",
    "loading",
    "confirmation",
    "pending",
  ])("does not signal or request blocked compaction: %s", async (blocked) => {
    if (blocked === "no-conversation") conversation = null;
    if (blocked === "acp")
      conversation = { ...activeConversation, agent_kind: "acp" };
    if (blocked === "running") agentState = AgentState.RUNNING;
    if (blocked === "loading") agentState = AgentState.LOADING;
    if (blocked === "confirmation")
      agentState = AgentState.AWAITING_USER_CONFIRMATION;
    const onRequested = vi.fn();
    unsubscribe.push(
      subscribeConversationContextChangeRequested(scope, onRequested),
    );
    const { result } = renderHook(() => useCompactContextAction(), { wrapper });

    if (blocked === "pending") {
      vi.mocked(
        AgentServerConversationService.condenseConversation,
      ).mockImplementation(() => new Promise(() => {}));
      await act(async () => {
        result.current.handleCompact();
        await vi.advanceTimersByTimeAsync(0);
      });
      onRequested.mockClear();
      vi.mocked(
        AgentServerConversationService.condenseConversation,
      ).mockClear();
    }
    expect(result.current.isDisabled).toBe(true);
    await act(async () => {
      result.current.handleCompact();
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(onRequested).not.toHaveBeenCalled();
    expect(
      AgentServerConversationService.condenseConversation,
    ).not.toHaveBeenCalled();
  });

  it("toasts counts when condensation lands and metrics drop", async () => {
    const { result } = renderHook(() => useCompactContextAction(50_000), {
      wrapper,
    });

    await act(async () => {
      result.current.handleCompact();
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(successToast).toHaveBeenCalledWith(
      I18nKey.CONVERSATION$COMPACT_CONTEXT_STARTED,
    );

    act(() => {
      useEventStore.getState().addEvent(condensationEvent("cond-x"));
      useMetricsStore.getState().setMetrics({
        cost: null,
        max_budget_per_task: null,
        usage: usage(20_000),
      });
    });

    expect(successToast).toHaveBeenCalledWith(
      I18nKey.CONVERSATION$COMPACT_CONTEXT_COMPLETE,
    );
    expect(errorToast).not.toHaveBeenCalled();
  });

  it("toasts error when no condensation arrives in time", async () => {
    const { result } = renderHook(() => useCompactContextAction(50_000), {
      wrapper,
    });

    await act(async () => {
      result.current.handleCompact();
      await vi.advanceTimersByTimeAsync(0);
    });
    act(() => {
      vi.advanceTimersByTime(90_000);
    });

    expect(errorToast).toHaveBeenCalledWith(
      I18nKey.CONVERSATION$COMPACT_CONTEXT_FAILED,
    );
  });
  it("shows one error for a rejected compaction and allows another attempt", async () => {
    const reason = "This agent has no condenser configured";
    vi.mocked(
      AgentServerConversationService.condenseConversation,
    ).mockRejectedValue(new HttpError(409, "Conflict", { detail: reason }));
    const { result } = renderHook(() => useCompactContextAction(50_000), {
      wrapper,
    });
    await act(async () => {
      result.current.handleCompact();
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(errorToast).toHaveBeenCalledExactlyOnceWith(reason);
    expect(successToast).not.toHaveBeenCalled();
    expect(result.current.isCompacting).toBe(false);
    expect(result.current.isDisabled).toBe(false);
  });
});
