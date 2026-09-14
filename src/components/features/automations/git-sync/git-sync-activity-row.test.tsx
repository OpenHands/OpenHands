import { render, screen, act } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { GitSyncActivityRow } from "./git-sync-activity-row";

// The global test mock returns the key and drops interpolation values; these
// tests are about the interpolated elapsed time, so keep it.
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, unknown>) =>
      options ? `${key}|${JSON.stringify(options)}` : key,
    i18n: { language: "en", exists: () => false },
  }),
}));

const STARTED_AT = "2026-08-12T12:00:00Z";

afterEach(() => {
  vi.useRealTimers();
});

describe("GitSyncActivityRow", () => {
  it("counts the elapsed time up while the cycle runs", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(STARTED_AT));

    render(
      <GitSyncActivityRow
        state="running"
        startedAt={STARTED_AT}
        pendingCount={0}
      />,
    );

    const row = screen.getByTestId("git-sync-activity-row");
    expect(row).toHaveTextContent('"time":"0s"');

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3_000);
    });

    expect(row).toHaveTextContent('"time":"3s"');
  });

  it("stops ticking once the cycle is no longer running", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(STARTED_AT));

    const { rerender, unmount } = render(
      <GitSyncActivityRow
        state="running"
        startedAt={STARTED_AT}
        pendingCount={0}
      />,
    );
    expect(vi.getTimerCount()).toBeGreaterThan(0);

    rerender(
      <GitSyncActivityRow
        state="succeeded"
        startedAt={null}
        pendingCount={0}
      />,
    );
    expect(vi.getTimerCount()).toBe(0);

    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });
});
