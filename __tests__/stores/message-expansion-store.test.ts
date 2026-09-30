import { describe, expect, it } from "vitest";
import {
  expandedRowKey,
  messageExpansionKey,
  useMessageExpansionStore,
} from "#/stores/message-expansion-store";

describe("message expansion store pruning", () => {
  it("keeps an entry whose row is still in the history", () => {
    const { setExpanded, prune } = useMessageExpansionStore.getState();
    setExpanded(messageExpansionKey("single-message-0", "details"), true);

    // The prune set carries only row keys, so the `::details` suffix must not
    // disqualify the entry — otherwise every expansion is dropped whenever the
    // rendered item list changes (a streamed append, history pagination).
    prune(new Set(["single-message-0"]));

    expect(useMessageExpansionStore.getState().expanded).toEqual({
      "single-message-0::details": true,
    });
  });

  it("drops an entry whose row left the history", () => {
    const { setExpanded, prune } = useMessageExpansionStore.getState();
    setExpanded(messageExpansionKey("single-message-9", "details"), true);

    prune(new Set(["single-message-0"]));

    expect(useMessageExpansionStore.getState().expanded).toEqual({});
  });
});

describe("expandedRowKey", () => {
  it("returns the row part of a control key", () => {
    expect(expandedRowKey("group-a::details")).toBe("group-a");
    expect(expandedRowKey("group-a::model-1")).toBe("group-a");
  });

  it("passes through a key without a control suffix", () => {
    expect(expandedRowKey("group-a")).toBe("group-a");
  });
});
