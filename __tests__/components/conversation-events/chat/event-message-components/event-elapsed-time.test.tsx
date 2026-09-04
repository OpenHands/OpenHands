import { afterEach, describe, expect, it, vi } from "vitest";
import { act, screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { EventElapsedTime } from "#/components/conversation-events/chat/event-message-components/event-elapsed-time";

const BASE_TIME = new Date("2026-01-01T12:00:00.000Z");

afterEach(() => {
  vi.useRealTimers();
});

describe("EventElapsedTime", () => {
  // ── Completed (static) ──────────────────────────────────────────────────

  it("renders the static duration when endTimestamp is provided", () => {
    // 5 seconds between start and end.
    const start = "2026-01-01T12:00:00.000Z";
    const end = "2026-01-01T12:00:05.000Z";

    // Pin the wall clock so formatTimeDelta doesn't drift during the test.
    vi.useFakeTimers();
    vi.setSystemTime(new Date(end));

    renderWithProviders(
      <EventElapsedTime startTimestamp={start} endTimestamp={end} />,
    );

    expect(screen.getByTestId("event-elapsed-time")).toHaveTextContent("5s");
  });

  it("does not start a live ticker when endTimestamp is provided", async () => {
    vi.useFakeTimers();
    const start = "2026-01-01T12:00:00.000Z";
    const end = "2026-01-01T12:00:05.000Z";
    vi.setSystemTime(new Date(end));

    renderWithProviders(
      <EventElapsedTime startTimestamp={start} endTimestamp={end} />,
    );

    const textBefore = screen.getByTestId("event-elapsed-time").textContent;

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3_000);
    });

    // Static display — the text should be the same even after 3 s have passed.
    expect(screen.getByTestId("event-elapsed-time").textContent).toBe(
      textBefore,
    );
  });

  // ── Running (live) ───────────────────────────────────────────────────────

  it("renders an initial live elapsed duration when only startTimestamp is given", () => {
    vi.useFakeTimers();
    vi.setSystemTime(BASE_TIME);

    // Start was 3 seconds ago from the pinned clock.
    const start = new Date(BASE_TIME.getTime() - 3_000).toISOString();

    renderWithProviders(<EventElapsedTime startTimestamp={start} />);

    expect(screen.getByTestId("event-elapsed-time")).toHaveTextContent("3s");
  });

  it("updates the live counter after each elapsed second", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(BASE_TIME);

    // Action started right now (0 s elapsed).
    const start = BASE_TIME.toISOString();

    renderWithProviders(<EventElapsedTime startTimestamp={start} />);

    expect(screen.getByTestId("event-elapsed-time")).toHaveTextContent("0s");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000);
    });
    expect(screen.getByTestId("event-elapsed-time")).toHaveTextContent("1s");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000);
    });
    expect(screen.getByTestId("event-elapsed-time")).toHaveTextContent("3s");
  });

  it("stops ticking when endTimestamp is supplied after initial render", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(BASE_TIME);

    const start = BASE_TIME.toISOString();
    const end = new Date(BASE_TIME.getTime() + 4_000).toISOString();

    const { rerender } = renderWithProviders(
      <EventElapsedTime startTimestamp={start} />,
    );

    // Advance 4 s — live counter should update.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(4_000);
    });
    expect(screen.getByTestId("event-elapsed-time")).toHaveTextContent("4s");

    // Observation arrived — switch to static mode.
    rerender(<EventElapsedTime startTimestamp={start} endTimestamp={end} />);

    const textAfterCompletion =
      screen.getByTestId("event-elapsed-time").textContent;

    // Advance another 3 s — no more ticking should occur.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3_000);
    });
    expect(screen.getByTestId("event-elapsed-time").textContent).toBe(
      textAfterCompletion,
    );
  });

  // ── Edge cases ───────────────────────────────────────────────────────────

  it("clamps a negative duration to 0s (clock skew)", () => {
    vi.useFakeTimers();
    vi.setSystemTime(BASE_TIME);

    // Start timestamp is 2 seconds in the future relative to "now".
    const futureStart = new Date(
      BASE_TIME.getTime() + 2_000,
    ).toISOString();
    const futureEnd = new Date(BASE_TIME.getTime() + 5_000).toISOString();

    renderWithProviders(
      <EventElapsedTime
        startTimestamp={futureStart}
        endTimestamp={futureEnd}
      />,
    );

    // endMs - startMs = 3 s, which is positive, so this tests a completed case.
    expect(screen.getByTestId("event-elapsed-time")).toHaveTextContent("3s");
  });

  it("clamps a negative live duration to 0s when start is in the future", () => {
    vi.useFakeTimers();
    vi.setSystemTime(BASE_TIME);

    // Start is 5 seconds ahead of the client clock.
    const futureStart = new Date(
      BASE_TIME.getTime() + 5_000,
    ).toISOString();

    renderWithProviders(<EventElapsedTime startTimestamp={futureStart} />);

    // Negative delta → clamped to 0 → formatTimeDelta returns "0s".
    expect(screen.getByTestId("event-elapsed-time")).toHaveTextContent("0s");
  });

  it("renders nothing for an invalid startTimestamp", () => {
    const { container } = renderWithProviders(
      <EventElapsedTime startTimestamp="not-a-date" />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing for an invalid endTimestamp", () => {
    const { container } = renderWithProviders(
      <EventElapsedTime
        startTimestamp="2026-01-01T12:00:00.000Z"
        endTimestamp="not-a-date"
      />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("uses a semantic <time> element", () => {
    vi.useFakeTimers();
    vi.setSystemTime(BASE_TIME);

    const start = "2026-01-01T12:00:00.000Z";
    const end = "2026-01-01T12:00:10.000Z";

    renderWithProviders(
      <EventElapsedTime startTimestamp={start} endTimestamp={end} />,
    );

    const el = screen.getByTestId("event-elapsed-time");
    expect(el.tagName.toLowerCase()).toBe("time");
  });
});
