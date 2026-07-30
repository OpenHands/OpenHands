import { beforeEach, describe, expect, it } from "vitest";
import { usePopoutStore } from "#/stores/popout-store";
import { useEventStore } from "#/stores/use-event-store";
import type { MessageEvent } from "#/types/agent-server/core";

describe("usePopoutStore", () => {
  beforeEach(() => {
    usePopoutStore.setState({ popouts: [] });
    useEventStore.getState().clearEvents();
  });

  it("opens a new expanded window", () => {
    usePopoutStore.getState().openPopout({
      conversationId: "fork-1",
      title: "Trip planning (branch)",
    });

    expect(usePopoutStore.getState().popouts).toEqual([
      expect.objectContaining({
        conversationId: "fork-1",
        title: "Trip planning (branch)",
        mode: "expanded",
      }),
    ]);
  });

  it("re-focuses an already-open window instead of duplicating it", () => {
    const store = usePopoutStore.getState();
    store.openPopout({ conversationId: "fork-1", title: "A" });
    store.minimizePopout("fork-1");
    store.openPopout({ conversationId: "fork-1", title: "A" });

    const { popouts } = usePopoutStore.getState();
    expect(popouts).toHaveLength(1);
    expect(popouts[0].mode).toBe("expanded");
  });

  it("minimizes and expands a window", () => {
    const store = usePopoutStore.getState();
    store.openPopout({ conversationId: "fork-1", title: "A" });
    store.minimizePopout("fork-1");
    expect(usePopoutStore.getState().popouts[0].mode).toBe("minimized");
    store.expandPopout("fork-1");
    expect(usePopoutStore.getState().popouts[0].mode).toBe("expanded");
  });

  it("closes a window and clears its event bucket when it is not the primary route", () => {
    useEventStore.getState().addEvent("fork-1", {
      id: "evt-1",
      timestamp: "2024-01-01T00:00:00Z",
      source: "user",
      llm_message: {
        role: "user",
        content: [{ type: "text", text: "hi" }],
      },
      activated_microagents: [],
      extended_content: [],
    } as MessageEvent);

    usePopoutStore.getState().openPopout({
      conversationId: "fork-1",
      title: "A",
    });
    usePopoutStore.getState().closePopout("fork-1");

    expect(usePopoutStore.getState().popouts).toEqual([]);
    expect(useEventStore.getState().byConversation["fork-1"]).toBeUndefined();
  });
});
