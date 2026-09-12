import { beforeEach, describe, expect, it, vi } from "vitest";
import { useBtwStore } from "#/stores/btw-store";

const CONV_A = "conv-a";
const CONV_B = "conv-b";

const entriesFor = (conv: string) =>
  useBtwStore.getState().entriesByConversation[conv] ?? [];

describe("btw store", () => {
  beforeEach(() => {
    useBtwStore.setState({ entriesByConversation: {} });
  });

  it("adds a pending entry scoped to the given conversation", () => {
    const id = useBtwStore.getState().addPending(CONV_A, "why?");
    expect(entriesFor(CONV_A)).toEqual([
      { id, question: "why?", status: "pending" },
    ]);
    expect(entriesFor(CONV_B)).toEqual([]);
  });

  it("resolve and fail update status and response", () => {
    const id = useBtwStore.getState().addPending(CONV_A, "why?");
    const otherId = useBtwStore.getState().addPending(CONV_A, "what?");
    useBtwStore.getState().resolve(CONV_A, id, "because");
    expect(entriesFor(CONV_A)[0]).toMatchObject({
      status: "done",
      response: "because",
    });
    useBtwStore.getState().fail(CONV_A, id, "boom");
    expect(entriesFor(CONV_A)[0]).toMatchObject({
      status: "error",
      response: "boom",
    });
    expect(entriesFor(CONV_A)[1]).toEqual({
      id: otherId,
      question: "what?",
      status: "pending",
    });
  });

  it("dismiss removes only the targeted entry in the scoped conversation", () => {
    const aId = useBtwStore.getState().addPending(CONV_A, "qa");
    const remainingId = useBtwStore.getState().addPending(CONV_A, "still here");
    useBtwStore.getState().addPending(CONV_B, "qb");
    useBtwStore.getState().dismiss(CONV_A, aId);
    expect(entriesFor(CONV_A)).toEqual([
      {
        id: remainingId,
        question: "still here",
        status: "pending",
      },
    ]);
    expect(entriesFor(CONV_B)).toHaveLength(1);
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
      const { useBtwStore: freshStore } = await import("#/stores/btw-store");

      expect(connect).toHaveBeenCalledWith({ name: "BtwStore" });
      expect(freshStore.getState().entriesByConversation).toEqual({});

      const dismissedId = freshStore
        .getState()
        .addPending(CONV_A, "dismiss me");
      const resolvedId = freshStore.getState().addPending(CONV_A, "resolve me");
      const failedId = freshStore.getState().addPending(CONV_A, "fail me");
      freshStore.getState().resolve(CONV_A, resolvedId, "resolved");
      freshStore.getState().fail(CONV_A, failedId, "failed");
      freshStore.getState().dismiss(CONV_A, dismissedId);

      expect(freshStore.getState().entriesByConversation[CONV_A]).toEqual([
        {
          id: resolvedId,
          question: "resolve me",
          response: "resolved",
          status: "done",
        },
        {
          id: failedId,
          question: "fail me",
          response: "failed",
          status: "error",
        },
      ]);
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
