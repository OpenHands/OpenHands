import { describe, it, expect } from "vitest";
import { ActionEvent } from "#/types/agent-server/core";
import {
  mergeReasoning,
  splitActionNarration,
  splitInlineThink,
} from "#/components/conversation-events/chat/event-thought-helpers";

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

describe("mergeReasoning", () => {
  it("drops an exact duplicate fragment", () => {
    expect(mergeReasoning("Check the directory", "Check the directory")).toBe(
      "Check the directory",
    );
  });

  it("joins distinct fragments and skips blanks", () => {
    expect(mergeReasoning("First", "", "   ", "Second")).toBe(
      "First\n\nSecond",
    );
  });
});

describe("splitActionNarration", () => {
  const action = (overrides: Record<string, unknown>) =>
    ({
      thought: [],
      thinking_blocks: [],
      ...overrides,
    }) as unknown as ActionEvent;

  it("merges explicit reasoning with an identical inline block once", () => {
    const event = action({
      reasoning_content: "Check the directory",
      thought: [
        {
          type: "text",
          text: "<think>Check the directory</think>\nRunning ls",
        },
      ],
    });
    expect(splitActionNarration(event)).toEqual({
      reasoning: "Check the directory",
      message: "Running ls",
    });
  });

  it("keeps distinct explicit and inline reasoning as two fragments", () => {
    const event = action({
      reasoning_content: "Explicit reasoning",
      thought: [
        { type: "text", text: "<think>Inline reasoning</think>\nDo it" },
      ],
    });
    expect(splitActionNarration(event)).toEqual({
      reasoning: "Explicit reasoning\n\nInline reasoning",
      message: "Do it",
    });
  });

  it("falls back to explicit reasoning when there is no inline block", () => {
    const event = action({
      reasoning_content: "Only explicit",
      thought: [{ type: "text", text: "Plain message" }],
    });
    expect(splitActionNarration(event)).toEqual({
      reasoning: "Only explicit",
      message: "Plain message",
    });
  });
});
