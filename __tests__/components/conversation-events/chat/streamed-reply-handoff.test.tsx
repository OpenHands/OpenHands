import { act, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { Messages } from "#/components/conversation-events/chat/messages";
import { MessageEvent } from "#/types/agent-server/core";
import { StreamingDeltaEvent } from "#/types/agent-server/core/events/streaming-delta-event";

const slot = (content: string): StreamingDeltaEvent => ({
  id: "item-1",
  timestamp: "2026-06-12T12:00:00Z",
  source: "agent",
  kind: "StreamingDeltaEvent",
  content,
  reasoning_content: null,
});

// The durable message carries the slot's id and replaces it in place.
const finished = (text: string): MessageEvent => ({
  id: "item-1",
  timestamp: "2026-06-12T12:00:00Z",
  source: "agent",
  llm_message: { role: "assistant", content: [{ type: "text", text }] },
  activated_skills: [],
  extended_content: [],
});

describe("a streamed reply finishing", () => {
  let now = 0;
  let callbacks: Map<number, FrameRequestCallback>;
  const tick = (count: number) => {
    for (let i = 0; i < count; i += 1) {
      now += 16;
      const scheduled = [...callbacks.values()];
      callbacks.clear();
      act(() => scheduled.forEach((callback) => callback(now)));
    }
  };

  beforeEach(() => {
    now = 0;
    callbacks = new Map();
    let nextHandle = 1;
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
      const handle = nextHandle;
      nextHandle += 1;
      callbacks.set(handle, callback);
      return handle;
    });
    vi.stubGlobal("cancelAnimationFrame", (handle: number) => {
      callbacks.delete(handle);
    });
    vi.spyOn(performance, "now").mockImplementation(() => now);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("keeps revealing where the slot left off instead of jumping", () => {
    const text = `Hello${" world".repeat(30)}`;
    const { rerender } = renderWithProviders(
      <Messages messages={[slot("Hello")]} allEvents={[slot("Hello")]} />,
    );

    rerender(
      <Messages messages={[finished(text)]} allEvents={[finished(text)]} />,
    );
    tick(10);
    const bubble = () => screen.getByTestId("agent-message");
    const shown = bubble().textContent ?? "";
    expect(shown).toContain("Hello world");
    expect(shown).not.toContain(text);

    // ~185 characters at the reveal pace, then the finished message renders.
    tick(240);
    expect(bubble()).toHaveTextContent(text);
    expect(bubble().querySelector("[data-stream-word]")).toBeNull();
  });

  it("shows a message that was never streamed at once", () => {
    const text = `Hello${" world".repeat(30)}`;
    renderWithProviders(
      <Messages messages={[finished(text)]} allEvents={[finished(text)]} />,
    );
    expect(screen.getByTestId("agent-message")).toHaveTextContent(text);
  });
});
