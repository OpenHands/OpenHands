import type { ReactNode } from "react";
import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { Messages } from "#/components/conversation-events/chat/messages";
import type { MessageEvent } from "#/types/agent-server/core";
import type { StreamingDeltaEvent } from "#/types/agent-server/core/events/streaming-delta-event";

const { recordSyntaxHighlight } = vi.hoisted(() => ({
  recordSyntaxHighlight: vi.fn(),
}));

vi.mock("#/components/features/markdown/syntax-highlighter", () => ({
  SyntaxHighlighter: ({ children }: { children?: ReactNode }) => {
    recordSyntaxHighlight();
    return <code>{children}</code>;
  },
}));

vi.mock(
  "#/components/shared/buttons/conversation-confirmation-buttons",
  () => ({
    ConversationConfirmationButtons: () => (
      <span data-testid="confirmation-marker" />
    ),
  }),
);

const makeAgentMessage = (id: string, text: string): MessageEvent => ({
  id,
  timestamp: "2026-08-30T00:00:00Z",
  source: "agent",
  llm_message: {
    role: "assistant",
    content: [{ type: "text", text }],
  },
  activated_skills: [],
  extended_content: [],
});

const makeStreamingDelta = (content: string): StreamingDeltaEvent => ({
  id: "live-stream",
  timestamp: "2026-08-30T00:00:01Z",
  source: "agent",
  kind: "StreamingDeltaEvent",
  content,
  reasoning_content: null,
});

const codeHeavyMessage = (index: number) =>
  [
    `Historical response ${index}`,
    "```javascript",
    ...Array.from(
      { length: 50 },
      (_, line) => `const value${line} = ${index + line};`,
    ),
    "```",
  ].join("\n");

describe("Messages markdown render stability", () => {
  beforeEach(() => {
    recordSyntaxHighlight.mockClear();
  });

  it("does not re-render historical code blocks when the live tail changes", () => {
    const history = Array.from({ length: 20 }, (_, index) =>
      makeAgentMessage(`history-${index}`, codeHeavyMessage(index)),
    );

    const { rerender } = renderWithProviders(
      <Messages messages={history} allEvents={history} />,
    );

    expect(recordSyntaxHighlight).toHaveBeenCalledTimes(history.length);
    expect(
      screen
        .getByTestId("confirmation-marker")
        .closest('[data-testid="agent-message"]'),
    ).toHaveTextContent("Historical response 19");
    recordSyntaxHighlight.mockClear();

    const durableTail = makeAgentMessage("durable-tail", "Plain tail message");
    const withDurableTail = [...history, durableTail];
    rerender(
      <Messages messages={withDurableTail} allEvents={withDurableTail} />,
    );
    expect(
      screen
        .getByTestId("confirmation-marker")
        .closest('[data-testid="agent-message"]'),
    ).toHaveTextContent("Plain tail message");

    const firstStreamingTail = makeStreamingDelta("Live update 0");
    rerender(
      <Messages
        messages={[...withDurableTail, firstStreamingTail]}
        allEvents={[...withDurableTail, firstStreamingTail]}
      />,
    );

    for (let update = 1; update <= 5; update += 1) {
      const streamingTail = makeStreamingDelta(`Live update ${update}`);
      rerender(
        <Messages
          messages={[...withDurableTail, streamingTail]}
          allEvents={[...withDurableTail, streamingTail]}
        />,
      );
    }

    expect(screen.getByText("Live update 5")).toBeInTheDocument();
    expect(recordSyntaxHighlight).not.toHaveBeenCalled();
  });
});
