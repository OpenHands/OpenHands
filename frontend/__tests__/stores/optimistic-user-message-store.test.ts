import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";

// Reset the store between tests so state from one test does not leak into
// the next (Zustand stores are global singletons by default).
afterEach(() => {
  useOptimisticUserMessageStore.getState().removeOptimisticUserMessage();
});

describe("useOptimisticUserMessageStore", () => {
  it("should have null message and not-pending as initial state", () => {
    const { result } = renderHook(() => useOptimisticUserMessageStore());
    expect(result.current.optimisticUserMessage).toBeNull();
    expect(result.current.isPendingDelivery).toBe(false);
    expect(result.current.getOptimisticUserMessage()).toBeNull();
    expect(result.current.isOptimisticUserMessagePending()).toBe(false);
  });

  it("should default isPendingDelivery to false when not provided", () => {
    const { result } = renderHook(() => useOptimisticUserMessageStore());

    act(() => {
      result.current.setOptimisticUserMessage("Hello");
    });

    expect(result.current.optimisticUserMessage).toBe("Hello");
    expect(result.current.isPendingDelivery).toBe(false);
  });

  it("should mark message as pending when isPendingDelivery=true (issue #14181)", () => {
    const { result } = renderHook(() => useOptimisticUserMessageStore());

    act(() => {
      result.current.setOptimisticUserMessage("Queued hello", true);
    });

    expect(result.current.optimisticUserMessage).toBe("Queued hello");
    expect(result.current.isPendingDelivery).toBe(true);
    expect(result.current.isOptimisticUserMessagePending()).toBe(true);
  });

  it("should clear pending flag and message on remove", () => {
    const { result } = renderHook(() => useOptimisticUserMessageStore());

    act(() => {
      result.current.setOptimisticUserMessage("Queued", true);
    });
    expect(result.current.isPendingDelivery).toBe(true);

    act(() => {
      result.current.removeOptimisticUserMessage();
    });

    expect(result.current.optimisticUserMessage).toBeNull();
    expect(result.current.isPendingDelivery).toBe(false);
  });

  it("should clear the pending flag when the same message is replaced without the flag", () => {
    const { result } = renderHook(() => useOptimisticUserMessageStore());

    act(() => {
      result.current.setOptimisticUserMessage("Queued", true);
    });
    expect(result.current.isPendingDelivery).toBe(true);

    act(() => {
      result.current.setOptimisticUserMessage("Queued");
    });

    // Replacing without an explicit flag must reset the pending state — this
    // is what makes the indicator disappear when a queued message gets
    // re-set as a normal optimistic message after delivery.
    expect(result.current.isPendingDelivery).toBe(false);
  });
});
