import { describe, it, expect } from "vitest";
import {
  getMessageReasoningContent,
  splitInlineThink,
} from "#/components/conversation-events/chat/event-thought-helpers";
import type { MessageEvent } from "#/types/agent-server/core";

const makeMessageEvent = (
  llm_message: MessageEvent["llm_message"],
): MessageEvent =>
  ({
    id: "evt-1",
    source: "agent",
    timestamp: "2024-01-01T00:00:00.000Z",
    llm_message,
  }) as unknown as MessageEvent;

describe("getMessageReasoningContent", () => {
  it("returns llm_message.reasoning_content when present", () => {
    const event = makeMessageEvent({
      role: "assistant",
      content: [{ type: "text", text: "Hello" }],
      reasoning_content: "The user wants a greeting.",
    });

    expect(getMessageReasoningContent(event)).toBe(
      "The user wants a greeting.",
    );
  });

  it("falls back to thinking_blocks when reasoning_content is absent", () => {
    const event = makeMessageEvent({
      role: "assistant",
      content: [{ type: "text", text: "Hello" }],
      thinking_blocks: [
        {
          type: "thinking",
          signature: "sig",
          thinking: "Step one: parse the request.",
        },
        {
          type: "thinking",
          signature: "sig2",
          thinking: "Step two: compose the reply.",
        },
      ],
    });

    expect(getMessageReasoningContent(event)).toBe(
      "Step one: parse the request.\n\nStep two: compose the reply.",
    );
  });

  it("ignores redacted thinking blocks", () => {
    const event = makeMessageEvent({
      role: "assistant",
      content: [{ type: "text", text: "Hello" }],
      thinking_blocks: [{ type: "redacted_thinking", data: "…" }],
    });

    expect(getMessageReasoningContent(event)).toBe("");
  });

  it("returns an empty string when neither field is present", () => {
    const event = makeMessageEvent({
      role: "assistant",
      content: [{ type: "text", text: "Hello" }],
    });

    expect(getMessageReasoningContent(event)).toBe("");
  });

  it("treats null reasoning_content as absent", () => {
    const event = makeMessageEvent({
      role: "assistant",
      content: [{ type: "text", text: "Hello" }],
      reasoning_content: null,
    });

    expect(getMessageReasoningContent(event)).toBe("");
  });
});

describe("splitInlineThink", () => {
  it("returns content unchanged when there is no <think> block", () => {
    expect(splitInlineThink("Hello! How can I help?")).toEqual({
      reasoning: "",
      message: "Hello! How can I help?",
    });
  });

  it("extracts a leading closed <think> block and keeps the trailing message", () => {
    const content =
      "<think>The user wants a greeting. Simple.</think>\n\n\nHello!";
    expect(splitInlineThink(content)).toEqual({
      reasoning: "The user wants a greeting. Simple.",
      message: "Hello!",
    });
  });

  it("handles leading whitespace before the <think> block", () => {
    expect(splitInlineThink("\n  <think>thinking</think>\n\nHi")).toEqual({
      reasoning: "thinking",
      message: "Hi",
    });
  });

  it("returns an empty message when the content is reasoning only", () => {
    expect(splitInlineThink("<think>just thinking</think>")).toEqual({
      reasoning: "just thinking",
      message: "",
    });
  });

  // Regression: an agent that literally emits "<think>" as its finalized
  // answer (no closing tag) must render verbatim, not vanish into Thinking.
  it("renders a finalized unclosed leading <think> verbatim", () => {
    expect(splitInlineThink("<think>")).toEqual({
      reasoning: "",
      message: "<think>",
    });
  });

  // Mid-stream, an unclosed leading <think> is reasoning still arriving, so
  // it is hidden until </think> shows up.
  it("treats an unclosed leading <think> as reasoning while streaming", () => {
    expect(
      splitInlineThink("<think>The user is asking me to", { streaming: true }),
    ).toEqual({ reasoning: "The user is asking me to", message: "" });
  });

  // Only the first block is peeled; a <think> that is part of the answer
  // (after </think>) is preserved verbatim.
  it("preserves a <think> that appears after the reasoning block", () => {
    expect(splitInlineThink("<think>reasoning</think>\n\n<think>")).toEqual({
      reasoning: "reasoning",
      message: "<think>",
    });
  });

  // Regression (reviewer): a <think> that is NOT at the start must be
  // preserved verbatim — only litellm's leading reasoning block is extracted.
  it("leaves a mid-message <think> reference untouched", () => {
    const content = "You can wrap reasoning in <think> and </think> tags.";
    expect(splitInlineThink(content)).toEqual({
      reasoning: "",
      message: content,
    });
  });

  it("leaves an unclosed mid-message <think> untouched", () => {
    const content = "See the <think> tag for details.";
    expect(splitInlineThink(content)).toEqual({
      reasoning: "",
      message: content,
    });
  });

  it("does not extract a <think> block that follows real message text", () => {
    const content = "Here is an example: <think>not reasoning</think> done.";
    expect(splitInlineThink(content)).toEqual({
      reasoning: "",
      message: content,
    });
  });
});
