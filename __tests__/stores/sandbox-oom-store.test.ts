import { beforeEach, describe, expect, it } from "vitest";
import { useSandboxOOMStore } from "#/stores/sandbox-oom-store";

describe("sandbox-oom-store", () => {
  beforeEach(() => {
    useSandboxOOMStore.getState().clearAll();
  });

  it("starts with empty dismissedEventIds", () => {
    const { dismissedEventIds } = useSandboxOOMStore.getState();
    expect(dismissedEventIds).toEqual({});
  });

  it("records dismissed event IDs correctly", () => {
    useSandboxOOMStore.getState().dismiss(["evt-1", "evt-2"]);
    const { dismissedEventIds } = useSandboxOOMStore.getState();
    expect(dismissedEventIds["evt-1"]).toBe(true);
    expect(dismissedEventIds["evt-2"]).toBe(true);
    expect(dismissedEventIds["evt-3"]).toBeUndefined();
  });

  it("merges new dismissals with existing ones", () => {
    useSandboxOOMStore.getState().dismiss(["evt-1"]);
    useSandboxOOMStore.getState().dismiss(["evt-2"]);
    const { dismissedEventIds } = useSandboxOOMStore.getState();
    expect(dismissedEventIds["evt-1"]).toBe(true);
    expect(dismissedEventIds["evt-2"]).toBe(true);
  });

  it("clears all dismissals with clearAll", () => {
    useSandboxOOMStore.getState().dismiss(["evt-1"]);
    useSandboxOOMStore.getState().clearAll();
    const { dismissedEventIds } = useSandboxOOMStore.getState();
    expect(dismissedEventIds).toEqual({});
  });
});
