import { beforeEach, describe, expect, it } from "vitest";
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
  });

  it("dismiss removes only the targeted entry in the scoped conversation", () => {
    const aId = useBtwStore.getState().addPending(CONV_A, "qa");
    useBtwStore.getState().addPending(CONV_B, "qb");
    useBtwStore.getState().dismiss(CONV_A, aId);
    expect(entriesFor(CONV_A)).toEqual([]);
    expect(entriesFor(CONV_B)).toHaveLength(1);
  });
});
