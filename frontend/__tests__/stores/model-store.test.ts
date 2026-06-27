import { beforeEach, describe, expect, it } from "vitest";
import { useModelStore } from "#/stores/model-store";
import type { LlmProfileSummary } from "#/api/settings-service/profiles-service.api";

const CONV_A = "conv-a";
const CONV_B = "conv-b";

const profile = (name: string): LlmProfileSummary => ({
  name,
  model: "anthropic/claude-sonnet-4-6",
  base_url: null,
  api_key_set: true,
});

const entriesFor = (conv: string) =>
  useModelStore.getState().entriesByConversation[conv] ?? [];

const activeProfileFor = (conv: string) =>
  useModelStore.getState().activeProfileByConversation[conv] ?? null;

describe("model store", () => {
  beforeEach(() => {
    useModelStore.setState({
      entriesByConversation: {},
      activeProfileByConversation: {},
    });
  });

  it("show appends an entry scoped to the given conversation", () => {
    useModelStore.getState().show(CONV_A, null, [profile("default")]);
    expect(entriesFor(CONV_A)).toHaveLength(1);
    expect(entriesFor(CONV_A)[0]).toMatchObject({
      anchorEventId: null,
      profiles: [profile("default")],
    });
    expect(entriesFor(CONV_B)).toEqual([]);
  });

  it("show stacks multiple entries in invocation order, preserving anchors", () => {
    useModelStore.getState().show(CONV_A, null, [profile("a")]);
    useModelStore.getState().show(CONV_A, "evt-42", [profile("b")]);
    expect(entriesFor(CONV_A)).toHaveLength(2);
    expect(entriesFor(CONV_A)[0]).toMatchObject({
      anchorEventId: null,
      profiles: [profile("a")],
    });
    expect(entriesFor(CONV_A)[1]).toMatchObject({
      anchorEventId: "evt-42",
      profiles: [profile("b")],
    });
  });

  it("show keeps entries scoped per conversation", () => {
    useModelStore.getState().show(CONV_A, null, [profile("a")]);
    useModelStore.getState().show(CONV_B, null, [profile("b")]);
    expect(entriesFor(CONV_A)).toHaveLength(1);
    expect(entriesFor(CONV_B)).toHaveLength(1);
    expect(entriesFor(CONV_A)[0].profiles[0].name).toBe("a");
    expect(entriesFor(CONV_B)[0].profiles[0].name).toBe("b");
  });

  it("recordSwitch appends a switch entry tagged with switchedTo", () => {
    useModelStore.getState().recordSwitch(CONV_A, "evt-7", "gpt-5");
    expect(entriesFor(CONV_A)).toHaveLength(1);
    expect(entriesFor(CONV_A)[0]).toMatchObject({
      anchorEventId: "evt-7",
      profiles: [],
      switchedTo: "gpt-5",
    });
    expect(entriesFor(CONV_B)).toEqual([]);
  });

  it("recordSwitch and show stack into the same list in invocation order", () => {
    useModelStore.getState().show(CONV_A, "evt-1", [profile("default")]);
    useModelStore.getState().recordSwitch(CONV_A, "evt-2", "gpt-5");
    expect(entriesFor(CONV_A)).toHaveLength(2);
    expect(entriesFor(CONV_A)[0].switchedTo).toBeUndefined();
    expect(entriesFor(CONV_A)[0].profiles).toEqual([profile("default")]);
    expect(entriesFor(CONV_A)[1].switchedTo).toBe("gpt-5");
    expect(entriesFor(CONV_A)[1].profiles).toEqual([]);
  });

  it("recordSwitch tracks the active profile by name, per conversation, last-write-wins", () => {
    useModelStore.getState().recordSwitch(CONV_A, null, "gpt-5");
    expect(activeProfileFor(CONV_A)).toBe("gpt-5");
    expect(activeProfileFor(CONV_B)).toBeNull();

    // A later switch in the same conversation supersedes the previous one.
    useModelStore.getState().recordSwitch(CONV_A, null, "default");
    expect(activeProfileFor(CONV_A)).toBe("default");

    // Other conversations are unaffected.
    useModelStore.getState().recordSwitch(CONV_B, null, "claude");
    expect(activeProfileFor(CONV_A)).toBe("default");
    expect(activeProfileFor(CONV_B)).toBe("claude");
  });

  it("show does not change the tracked active profile", () => {
    useModelStore.getState().recordSwitch(CONV_A, null, "gpt-5");
    useModelStore.getState().show(CONV_A, null, [profile("default")]);
    expect(activeProfileFor(CONV_A)).toBe("gpt-5");
  });
});
