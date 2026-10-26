import { describe, expect, it } from "vitest";
import {
  collectWakeableConversations,
  isWakeableConversation,
  type WakeableConversation,
} from "./wakeable-conversations";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

const makeConversation = (
  overrides: Partial<WakeableConversation> = {},
): WakeableConversation => ({
  id: "conv-1",
  execution_status: ExecutionStatus.FINISHED,
  sandbox_id: null,
  sandbox_status: null,
  ...overrides,
});

describe("isWakeableConversation", () => {
  it("leaves running and finished conversations alone", () => {
    expect(
      isWakeableConversation(
        makeConversation({ execution_status: ExecutionStatus.RUNNING }),
      ),
    ).toBe(false);
    expect(isWakeableConversation(makeConversation())).toBe(false);
  });

  it("treats a paused execution as asleep", () => {
    expect(
      isWakeableConversation(
        makeConversation({ execution_status: ExecutionStatus.PAUSED }),
      ),
    ).toBe(true);
  });

  it("treats a paused sandbox as asleep regardless of execution state", () => {
    expect(
      isWakeableConversation(
        makeConversation({
          execution_status: ExecutionStatus.IDLE,
          sandbox_id: "sandbox-1",
          sandbox_status: "PAUSED",
        }),
      ),
    ).toBe(true);
  });

  it("never wakes a runtime that is gone or errored", () => {
    expect(
      isWakeableConversation(
        makeConversation({
          execution_status: ExecutionStatus.PAUSED,
          sandbox_status: "MISSING",
        }),
      ),
    ).toBe(false);
    expect(
      isWakeableConversation(
        makeConversation({
          execution_status: ExecutionStatus.PAUSED,
          sandbox_status: "ERROR",
        }),
      ),
    ).toBe(false);
  });
});

describe("collectWakeableConversations", () => {
  it("collects asleep conversations in list order, skipping archived ones", () => {
    const running = makeConversation({ id: "running" });
    const paused = makeConversation({
      id: "paused",
      execution_status: ExecutionStatus.PAUSED,
    });
    const sandboxPaused = makeConversation({
      id: "sandbox-paused",
      sandbox_id: "sandbox-9",
      sandbox_status: "PAUSED",
    });
    const archivedPaused = makeConversation({
      id: "archived-paused",
      execution_status: ExecutionStatus.PAUSED,
    });

    expect(
      collectWakeableConversations(
        [running, paused, sandboxPaused, archivedPaused],
        (conversation) => conversation.id === "archived-paused",
      ),
    ).toEqual([paused, sandboxPaused]);
  });
});
