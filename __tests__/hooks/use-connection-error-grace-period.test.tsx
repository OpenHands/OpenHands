import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useConnectionErrorGracePeriod } from "#/hooks/use-connection-error-grace-period";
import { useErrorMessageStore } from "#/stores/error-message-store";

beforeEach(() => {
  vi.useFakeTimers();
  useErrorMessageStore.getState().removeErrorMessage();
});
afterEach(() => vi.useRealTimers());

describe("connection recovery feedback", () => {
  it("stays quiet, then offers progress and a retry without declaring an agent failure", () => {
    const { result } = renderHook(() =>
      useConnectionErrorGracePeriod("runtime"),
    );
    act(() => result.current.report());
    act(() => vi.advanceTimersByTime(29999));
    expect(result.current.state).toBe("idle");
    act(() => vi.advanceTimersByTime(1));
    expect(result.current.state).toBe("reconnecting");
    act(() => result.current.report());
    act(() => vi.advanceTimersByTime(30000));
    expect(result.current.state).toBe("retryable");
    expect(useErrorMessageStore.getState().errorMessage).toBeNull();
  });

  it("clears an existing connection warning when a paused sandbox loses its URL", () => {
    const view = renderHook(
      ({ url }: { url: string | null }) => useConnectionErrorGracePeriod(url),
      {
        initialProps: { url: "runtime" as string | null },
      },
    );
    act(() => view.result.current.report());
    act(() => vi.advanceTimersByTime(30000));
    act(() =>
      useErrorMessageStore
        .getState()
        .setErrorMessage("old connection warning", "connection"),
    );
    view.rerender({ url: null });
    expect(view.result.current.state).toBe("idle");
    expect(useErrorMessageStore.getState().errorMessage).toBeNull();
    act(() => vi.advanceTimersByTime(60000));
    expect(view.result.current.state).toBe("idle");
  });

  it("cancels delayed feedback on recovery and leaves actionable errors intact", () => {
    const { result } = renderHook(() =>
      useConnectionErrorGracePeriod("runtime"),
    );
    act(() => result.current.report());
    act(() => vi.advanceTimersByTime(2000));
    act(() =>
      useErrorMessageStore
        .getState()
        .setErrorMessage("Fix your credentials", "conversation"),
    );
    act(() => result.current.cancel());
    act(() => vi.advanceTimersByTime(60000));
    expect(result.current.state).toBe("idle");
    expect(useErrorMessageStore.getState().errorMessage).toBe(
      "Fix your credentials",
    );
  });
});
