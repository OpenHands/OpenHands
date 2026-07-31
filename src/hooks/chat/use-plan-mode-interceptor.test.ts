import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { usePlanModeInterceptor } from "./use-plan-mode-interceptor";
import { AgentState } from "#/types/agent-state";

const setConversationMode = vi.fn();
const handlePlanClick = vi.fn();
let isCreatingConversation = false;

vi.mock("#/stores/conversation-store", () => ({
  useConversationStore: (selector: (s: unknown) => unknown) =>
    selector({ setConversationMode }),
}));
vi.mock("#/hooks/use-handle-plan-click", () => ({
  useHandlePlanClick: () => ({
    handlePlanClick,
    isCreatingConversation,
  }),
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
    expect(handlePlanClick).toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("switches back to code mode for /code", () => {
    const { intercept, onSubmit } = setup(CONV);
    intercept("/code");
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
});
