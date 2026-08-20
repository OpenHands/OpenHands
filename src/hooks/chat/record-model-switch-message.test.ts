import { describe, it, expect, beforeEach } from "vitest";
import { useModelStore } from "#/stores/model-store";
import { OpenHandsEvent } from "#/types/agent-server/core";
import { seedModelSwitchesFromHistory } from "./record-model-switch-message";

const userMessage = (id: string): OpenHandsEvent =>
  ({
    id,
    timestamp: "2024-01-01T00:00:00Z",
    source: "user",
    llm_message: { role: "user", content: [{ type: "text", text: "hi" }] },
  }) as unknown as OpenHandsEvent;

const switchObservation = (
  id: string,
  profileName: string,
  isError = false,
): OpenHandsEvent =>
  ({
    id,
    timestamp: "2024-01-01T00:00:00Z",
    source: "environment",
    action_id: `action-${id}`,
    observation: {
      kind: "SwitchLLMObservation",
      content: [],
      is_error: isError,
      profile_name: profileName,
      reason: null,
      active_model: null,
    },
  }) as unknown as OpenHandsEvent;

// An agent action event. `ThinkAction` is renderable (shown as a thinking
// block); `PlanningFileEditorAction` is hidden by `shouldRenderEvent`.
const agentAction = (id: string, kind: string): OpenHandsEvent =>
  ({
    id,
    timestamp: "2024-01-01T00:00:00Z",
    source: "agent",
    action: { kind, thought: "t" },
    tool_name: "tool",
    tool_call_id: `call-${id}`,
  }) as unknown as OpenHandsEvent;

const entriesFor = (conversationId: string) =>
  useModelStore.getState().entriesByConversation[conversationId] ?? [];

describe("seedModelSwitchesFromHistory", () => {
  beforeEach(() => {
    useModelStore.getState().clearAll();
  });

  it("seeds a successful switch anchored to the prior renderable event", () => {
    seedModelSwitchesFromHistory("c1", [
      userMessage("u1"),
      switchObservation("o1", "fast"),
    ]);

    const entries = entriesFor("c1");
    expect(entries).toHaveLength(1);
    expect(entries[0].switchedTo).toBe("fast");
    expect(entries[0].anchorEventId).toBe("u1");
    expect(entries[0].id).toBe("history-switch:o1");
  });

  it("is idempotent across re-seeds (e.g. reloads)", () => {
    const events = [userMessage("u1"), switchObservation("o1", "fast")];
    seedModelSwitchesFromHistory("c1", events);
    seedModelSwitchesFromHistory("c1", events);

    expect(entriesFor("c1")).toHaveLength(1);
  });

  it("ignores failed switches (they still render as error cards)", () => {
    seedModelSwitchesFromHistory("c1", [
      userMessage("u1"),
      switchObservation("e1", "fast", true),
    ]);

    expect(entriesFor("c1")).toHaveLength(0);
  });

  it("never anchors to a non-rendered event (must land on a rendered id)", () => {
    // PlanningFileEditorAction is hidden by shouldRenderEvent, so the renderer
    // never mounts it; anchoring there would orphan the message. The anchor
    // must fall back to the prior rendered event (the user message).
    seedModelSwitchesFromHistory("c1", [
      userMessage("u1"),
      agentAction("p1", "PlanningFileEditorAction"),
      switchObservation("o1", "fast"),
    ]);

    const entries = entriesFor("c1");
    expect(entries).toHaveLength(1);
    expect(entries[0].anchorEventId).toBe("u1");
  });

  it("anchors to a renderable ThinkAction that precedes the switch", () => {
    // In uiEvents the ThinkObservation is dropped and the ThinkAction is kept
    // (and rendered as a thinking block), so it is a valid anchor.
    seedModelSwitchesFromHistory("c1", [
      userMessage("u1"),
      agentAction("t1", "ThinkAction"),
      switchObservation("o1", "architect"),
    ]);

    const entries = entriesFor("c1");
    expect(entries).toHaveLength(1);
    expect(entries[0].anchorEventId).toBe("t1");
  });

  it("anchors to null when no renderable event precedes the switch", () => {
    seedModelSwitchesFromHistory("c1", [switchObservation("o1", "architect")]);

    const entries = entriesFor("c1");
    expect(entries).toHaveLength(1);
    expect(entries[0].anchorEventId).toBeNull();
  });

  it("preserves order and anchors for multiple switches", () => {
    seedModelSwitchesFromHistory("c1", [
      userMessage("u1"),
      switchObservation("o1", "fast"),
      userMessage("u2"),
      switchObservation("o2", "architect"),
    ]);

    const entries = entriesFor("c1");
    expect(entries).toHaveLength(2);
    expect(entries[0]).toMatchObject({
      switchedTo: "fast",
      anchorEventId: "u1",
    });
    expect(entries[1]).toMatchObject({
      switchedTo: "architect",
      anchorEventId: "u2",
    });
  });
});
