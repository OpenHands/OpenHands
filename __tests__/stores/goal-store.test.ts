import { beforeEach, describe, expect, it } from "vitest";
import { useGoalStore } from "#/stores/goal-store";
import { GoalStatus } from "#/types/agent-server/core/events/conversation-state-event";

const CONV_A = "conv-a";
const CONV_B = "conv-b";

const statusFor = (conv: string) =>
  useGoalStore.getState().statusByConversation[conv];

const makeStatus = (overrides: Partial<GoalStatus> = {}): GoalStatus => ({
  active: true,
  status: "running",
  iteration: 0,
  max_iterations: 10,
  objective: "make pytest pass",
  verdict: null,
  ...overrides,
});

describe("goal store", () => {
  beforeEach(() => {
    useGoalStore.setState({ statusByConversation: {} });
  });

  it("sets status scoped to the given conversation", () => {
    const status = makeStatus();
    useGoalStore.getState().setStatus(CONV_A, status);
    expect(statusFor(CONV_A)).toEqual(status);
    expect(statusFor(CONV_B)).toBeUndefined();
  });

  it("setStatus overwrites the previous status for a conversation", () => {
    useGoalStore.getState().setStatus(CONV_A, makeStatus());
    const next = makeStatus({
      active: false,
      status: "complete",
      iteration: 2,
      verdict: { score: 1, complete: true, missing: "" },
    });
    useGoalStore.getState().setStatus(CONV_A, next);
    expect(statusFor(CONV_A)).toEqual(next);
  });
});
