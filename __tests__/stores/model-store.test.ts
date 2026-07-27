import { beforeEach, describe, expect, it } from "vitest";
import { useModelStore } from "#/stores/model-store";
import type { ProfileInfo } from "#/api/profiles-service/profiles-service.api";

const CONV_A = "conv-a";
const CONV_B = "conv-b";

const entriesFor = (conversationId: string) =>
  useModelStore.getState().entriesByConversation[conversationId] ?? [];

const profile: ProfileInfo = {
  name: "haiku",
  model: "anthropic/claude-haiku-4-5",
  base_url: null,
  api_key_set: true,
};

describe("model store", () => {
  beforeEach(() => {
    useModelStore.setState({
      entriesByConversation: {},
      activeProfileByConversation: {},
    });
  });

  it("adds profile-list entries scoped to the conversation", () => {
    useModelStore.getState().show(CONV_A, "event-1", [profile]);

    expect(entriesFor(CONV_A)).toEqual([
      expect.objectContaining({
        id: expect.any(String),
        anchorEventId: "event-1",
        profiles: [profile],
      }),
    ]);
    expect(entriesFor(CONV_B)).toEqual([]);
  });

  it("records profile switches without mutating previous list entries", () => {
    useModelStore.getState().show(CONV_A, null, [profile]);
    useModelStore.getState().recordSwitch(CONV_A, "event-2", "gpt");

    expect(entriesFor(CONV_A)).toEqual([
      expect.objectContaining({
        anchorEventId: null,
        profiles: [profile],
      }),
      expect.objectContaining({
        anchorEventId: "event-2",
        profiles: [],
        switchedTo: "gpt",
      }),
    ]);
    // recordSwitch also tags this as the conversation's optimistic active
    // profile so the SwitchProfileButton reflects the new selection instantly.
    expect(useModelStore.getState().activeProfileByConversation[CONV_A]).toBe(
      "gpt",
    );
  });

  it("clearActiveProfile drops only the optimistic profile entry", () => {
    useModelStore.getState().show(CONV_A, "event-1", [profile]);
    useModelStore.getState().recordSwitch(CONV_A, "event-2", "gpt");

    useModelStore.getState().clearActiveProfile(CONV_A);

    expect(
      useModelStore.getState().activeProfileByConversation[CONV_A],
    ).toBeUndefined();
    // Chat-history entries for the conversation are preserved.
    expect(entriesFor(CONV_A)).toHaveLength(2);
  });

  it("clears entries for one conversation or all conversations", () => {
    useModelStore.getState().show(CONV_A, "event-1", [profile]);
    useModelStore.getState().recordSwitch(CONV_B, "event-2", "gpt");

    useModelStore.getState().clear(CONV_A);

    expect(entriesFor(CONV_A)).toEqual([]);
    expect(entriesFor(CONV_B)).toHaveLength(1);

    useModelStore.getState().clearAll();

    expect(entriesFor(CONV_B)).toEqual([]);
  });
});
