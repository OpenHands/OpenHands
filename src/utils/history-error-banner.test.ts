import { describe, it, expect } from "vitest";
import { shouldSeedHistoryErrorBanner } from "./history-error-banner";

import type { OpenHandsEvent } from "#/types/agent-server/core";

const makeError = (id: string): OpenHandsEvent =>
  ({
    id,
    kind: "ConversationErrorEvent",
    timestamp: `2026-10-04T13:00:0${id}:00.000Z`,
    source: "environment",
    code: "LLMServiceUnavailableError",
    detail: "litellm.InternalServerError: Connection error.",
    classification: { kind: "llm", retryable: false, user_action: "none" },
  }) as unknown as OpenHandsEvent;

const makeAgentMessage = (id: string): OpenHandsEvent =>
  ({
    id,
    kind: "MessageEvent",
    timestamp: `2026-10-04T13:00:0${id}:00.000Z`,
    source: "agent",
    llm_message: {
      role: "assistant",
      content: [{ type: "text", text: "done" }],
    },
  }) as unknown as OpenHandsEvent;

const makeUserMessage = (id: string): OpenHandsEvent =>
  ({
    id,
    kind: "MessageEvent",
    timestamp: `2026-10-04T13:00:0${id}:00.000Z`,
    source: "user",
    llm_message: { role: "user", content: [{ type: "text", text: "hi" }] },
  }) as unknown as OpenHandsEvent;

const makeFinishAction = (id: string): OpenHandsEvent =>
  ({
    id,
    kind: "ActionEvent",
    timestamp: `2026-10-04T13:00:0${id}:00.000Z`,
    source: "agent",
    action: { kind: "FinishAction", message: "done" },
    tool_name: "finish",
    tool_call_id: "call_1",
  }) as unknown as OpenHandsEvent;

describe("history error banner seeding", () => {
  it("seeds when the last outcome is an error", () => {
    const events = [makeUserMessage("1"), makeError("2")];
    expect(shouldSeedHistoryErrorBanner(events)).toBe(true);
  });

  it("stays quiet when the run recovered after the error", () => {
    const events = [
      makeUserMessage("1"),
      makeError("2"),
      makeAgentMessage("3"),
    ];
    expect(shouldSeedHistoryErrorBanner(events)).toBe(false);
  });

  it("stays quiet when the run finished after the error", () => {
    const events = [
      makeUserMessage("1"),
      makeError("2"),
      makeFinishAction("3"),
    ];
    expect(shouldSeedHistoryErrorBanner(events)).toBe(false);
  });

  it("stays quiet on a healthy conversation", () => {
    const events = [makeUserMessage("1"), makeAgentMessage("2")];
    expect(shouldSeedHistoryErrorBanner(events)).toBe(false);
  });

  it("stays quiet when there is no error at all", () => {
    const events = [makeUserMessage("1")];
    expect(shouldSeedHistoryErrorBanner(events)).toBe(false);
  });

  it("seeds when a new failure follows an earlier recovered run", () => {
    const events = [
      makeUserMessage("1"),
      makeError("2"),
      makeAgentMessage("3"),
      makeUserMessage("4"),
      makeError("5"),
    ];
    expect(shouldSeedHistoryErrorBanner(events)).toBe(true);
  });
});
