import { act, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { LiveElapsedClock } from "#/components/features/chat/live-elapsed-clock";

describe("LiveElapsedClock", () => {
  it("ticks up from the start timestamp", () => {
    vi.useFakeTimers();
    try {
      vi.setSystemTime(new Date("2026-11-02T12:00:00Z"));
      render(
        <LiveElapsedClock startTimestamp="2026-11-02T11:59:55Z" align="left" />,
      );
      expect(screen.getByTestId("live-elapsed-clock")).toHaveTextContent(
        "00:05",
      );

      act(() => {
        vi.advanceTimersByTime(5_000);
      });
      expect(screen.getByTestId("live-elapsed-clock")).toHaveTextContent(
        "00:10",
      );
    } finally {
      vi.useRealTimers();
    }
  });
});
