import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { usePlanModeInterceptor } from "./use-plan-mode-interceptor";
import { AgentState } from "#/types/agent-state";
import type { WebSocketConnectionState } from "#/contexts/conversation-websocket-context";

const setConversationMode = vi.fn();
const handlePlanClick = vi.fn();
let isCreatingConversation = false;
let hasPlanner = false;
let webSocketStatus: WebSocketConnectionState = "OPEN";

vi.mock("#/stores/conversation-store", () => ({
  useConversationStore: (selector: (s: unknown) => unknown) =>
    selector({ setConversationMode }),
}));
vi.mock("#/hooks/use-handle-plan-click", () => ({
  useHandlePlanClick: () => ({
    handlePlanClick,
    hasPlanner,
    isCreatingConversation,
  }),
}));
vi.mock("#/hooks/use-unified-websocket-status", () => ({
  useUnifiedWebSocketStatus: () => webSocketStatus,
}));

const CONV = "conv-1";

const setup = (
  conversationId: string | null,
  curAgentState: AgentState = AgentState.AWAITING_USER_INPUT,
) => {
  const onSubmit = vi.fn();
  const { result } = renderHook(() =>
    usePlanModeInterceptor(conversationId, curAgentState, onSubmit),
  );
  return { intercept: result.current, onSubmit };
};

describe("usePlanModeInterceptor", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    isCreatingConversation = false;
    hasPlanner = false;
    webSocketStatus = "OPEN";
  });

  it("passes a non-command message straight through to onSubmit", () => {
    const { intercept, onSubmit } = setup(CONV);
    intercept("hello there");
    expect(onSubmit).toHaveBeenCalledWith("hello there");
    expect(handlePlanClick).not.toHaveBeenCalled();
    expect(setConversationMode).not.toHaveBeenCalled();
  });

  it("enables plan mode for /plan", () => {
    const { intercept, onSubmit } = setup(CONV);
    intercept("/plan");
    expect(handlePlanClick).toHaveBeenCalledWith(undefined, undefined);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("creates the planner with the task as its initial message for /plan <task> when no planner exists yet", () => {
    const { intercept, onSubmit } = setup(CONV);
    intercept("/plan   Build a website about open source.  ");
    expect(handlePlanClick).toHaveBeenCalledWith(
      undefined,
      "Build a website about open source.",
    );
    expect(onSubmit).not.toHaveBeenCalled();
    expect(setConversationMode).not.toHaveBeenCalled();
  });

  it("sends the task straight to the existing planner for /plan <task> when a planner already exists", () => {
    hasPlanner = true;
    const { intercept, onSubmit } = setup(CONV);
    intercept("/plan Build a website about open source.");
    expect(setConversationMode).toHaveBeenCalledWith("plan");
    expect(onSubmit).toHaveBeenCalledWith("Build a website about open source.");
    expect(handlePlanClick).not.toHaveBeenCalled();
  });

  it("falls back to a bare toggle for /plan with only whitespace after it", () => {
    hasPlanner = true;
    const { intercept, onSubmit } = setup(CONV);
    intercept("/plan   ");
    expect(handlePlanClick).toHaveBeenCalledWith(undefined, undefined);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("switches back to code mode for /code", () => {
    const { intercept, onSubmit } = setup(CONV);
    intercept("/code");
    expect(setConversationMode).toHaveBeenCalledWith("code");
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("switches to code mode and sends the task for /code <task>", () => {
    const { intercept, onSubmit } = setup(CONV);
    intercept("/code   fix the bug in auth.ts  ");
    expect(setConversationMode).toHaveBeenCalledWith("code");
    expect(onSubmit).toHaveBeenCalledWith("fix the bug in auth.ts");
    expect(handlePlanClick).not.toHaveBeenCalled();
  });

  it("falls back to a bare toggle for /code with only whitespace after it", () => {
    const { intercept, onSubmit } = setup(CONV);
    intercept("/code   ");
    expect(setConversationMode).toHaveBeenCalledWith("code");
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("passes through (no toggle) when there is no conversation", () => {
    const { intercept, onSubmit } = setup(null);
    intercept("/plan");
    expect(onSubmit).toHaveBeenCalledWith("/plan");
    expect(handlePlanClick).not.toHaveBeenCalled();
  });

  it("swallows /plan while the agent is running (matches the disabled button)", () => {
    const { intercept, onSubmit } = setup(CONV, AgentState.RUNNING);
    intercept("/plan");
    expect(handlePlanClick).not.toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("swallows /code while the agent is running (matches the disabled button)", () => {
    const { intercept, onSubmit } = setup(CONV, AgentState.RUNNING);
    intercept("/code");
    expect(setConversationMode).not.toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("swallows /plan while a planning conversation is already being created", () => {
    isCreatingConversation = true;
    const { intercept, onSubmit } = setup(CONV);
    intercept("/plan");
    expect(handlePlanClick).not.toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("swallows /plan while the websocket is disconnected (matches the disabled button)", () => {
    webSocketStatus = "CLOSED";
    const { intercept, onSubmit } = setup(CONV);
    intercept("/plan");
    expect(handlePlanClick).not.toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("swallows /code while the websocket is disconnected (matches the disabled button)", () => {
    webSocketStatus = "CLOSED";
    const { intercept, onSubmit } = setup(CONV);
    intercept("/code");
    expect(setConversationMode).not.toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });
});
