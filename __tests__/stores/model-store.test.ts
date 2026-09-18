import { beforeEach, describe, expect, it, vi } from "vitest";
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
    useModelStore.getState().recordSwitch(CONV_B, "event-3", "haiku");

    useModelStore.getState().clearActiveProfile(CONV_A);

    expect(
      useModelStore.getState().activeProfileByConversation[CONV_A],
    ).toBeUndefined();
    expect(useModelStore.getState().activeProfileByConversation[CONV_B]).toBe(
      "haiku",
    );
    // Chat-history entries for the conversation are preserved.
    expect(entriesFor(CONV_A)).toHaveLength(2);
  });

  it("clears entries for one conversation or all conversations", () => {
    useModelStore.getState().show(CONV_A, "event-1", [profile]);
    useModelStore.getState().recordSwitch(CONV_B, "event-2", "gpt");

    useModelStore.getState().clear(CONV_A);

    expect(entriesFor(CONV_A)).toEqual([]);
    expect(entriesFor(CONV_B)).toHaveLength(1);
    expect(useModelStore.getState().activeProfileByConversation[CONV_B]).toBe(
      "gpt",
    );

    useModelStore.getState().clearAll();

    expect(entriesFor(CONV_B)).toEqual([]);
    expect(useModelStore.getState().activeProfileByConversation).toEqual({});
  });

  it("seeds historical switches once while preserving live entries", () => {
    useModelStore
      .getState()
      .seedSwitches(CONV_B, [
        { id: "first", anchorEventId: null, profileName: "seeded" },
      ]);
    expect(entriesFor(CONV_B)).toHaveLength(1);

    useModelStore.getState().show(CONV_A, "live-event", [profile]);
    const liveId = entriesFor(CONV_A)[0].id;
    const switches = [
      { id: "switch-1", anchorEventId: null, profileName: "haiku" },
      { id: "switch-2", anchorEventId: "event-2", profileName: "gpt" },
      { id: liveId, anchorEventId: "duplicate", profileName: "ignored" },
    ];

    useModelStore.getState().seedSwitches(CONV_A, switches);
    expect(entriesFor(CONV_A)).toEqual([
      expect.objectContaining({ id: liveId, profiles: [profile] }),
      {
        id: "switch-1",
        anchorEventId: null,
        profiles: [],
        switchedTo: "haiku",
      },
      {
        id: "switch-2",
        anchorEventId: "event-2",
        profiles: [],
        switchedTo: "gpt",
      },
    ]);

    const beforeDuplicateSeed = useModelStore.getState();
    useModelStore.getState().seedSwitches(CONV_A, switches);
    expect(useModelStore.getState()).toBe(beforeDuplicateSeed);
  });

  it("does nothing when clearing a missing optimistic profile", () => {
    const before = useModelStore.getState();
    useModelStore.getState().clearActiveProfile("missing");
    expect(useModelStore.getState()).toBe(before);
  });

  it("creates a complete fresh store with its named devtools connection", async () => {
    const connection = {
      init: vi.fn(),
      send: vi.fn(),
      subscribe: vi.fn(),
    };
    const connect = vi.fn(() => connection);
    const previousExtension = Object.getOwnPropertyDescriptor(
      window,
      "__REDUX_DEVTOOLS_EXTENSION__",
    );
    Object.defineProperty(window, "__REDUX_DEVTOOLS_EXTENSION__", {
      configurable: true,
      value: { connect },
    });
    vi.resetModules();

    try {
      const { useModelStore: freshStore } =
        await import("#/stores/model-store");

      expect(connect).toHaveBeenCalledWith({ name: "ModelStore" });
      expect(freshStore.getState().entriesByConversation).toEqual({});
      expect(freshStore.getState().activeProfileByConversation).toEqual({});

      freshStore.getState().show(CONV_A, "event-1", [profile]);
      freshStore.getState().recordSwitch(CONV_B, "event-2", "gpt");
      freshStore
        .getState()
        .seedSwitches(CONV_A, [
          { id: "seeded", anchorEventId: null, profileName: "haiku" },
        ]);
      expect(freshStore.getState().entriesByConversation[CONV_A]).toHaveLength(
        2,
      );
      expect(freshStore.getState().activeProfileByConversation[CONV_B]).toBe(
        "gpt",
      );

      freshStore.getState().clearActiveProfile(CONV_B);
      expect(freshStore.getState().activeProfileByConversation).toEqual({});

      freshStore.getState().clear(CONV_A);
      expect(
        freshStore.getState().entriesByConversation[CONV_A],
      ).toBeUndefined();
      expect(freshStore.getState().entriesByConversation[CONV_B]).toHaveLength(
        1,
      );

      freshStore.getState().clearAll();
      expect(freshStore.getState().entriesByConversation).toEqual({});
      expect(freshStore.getState().activeProfileByConversation).toEqual({});
      freshStore.devtools?.cleanup();
    } finally {
      if (previousExtension) {
        Object.defineProperty(
          window,
          "__REDUX_DEVTOOLS_EXTENSION__",
          previousExtension,
        );
      } else {
        Reflect.deleteProperty(window, "__REDUX_DEVTOOLS_EXTENSION__");
      }
      vi.resetModules();
    }
  });
});
