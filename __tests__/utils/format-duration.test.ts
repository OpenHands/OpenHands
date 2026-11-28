import { describe, expect, it } from "vitest";
import { formatDurationMs } from "#/utils/format-duration";

describe("formatDurationMs", () => {
  it("formats sub-minute durations in whole seconds", () => {
    expect(formatDurationMs(0)).toBe("~0s");
    expect(formatDurationMs(500)).toBe("~1s");
    expect(formatDurationMs(12_000)).toBe("~12s");
    expect(formatDurationMs(59_000)).toBe("~59s");
  });

  it("formats minute-and-second durations", () => {
    expect(formatDurationMs(60_000)).toBe("~1m");
    expect(formatDurationMs(65_000)).toBe("~1m 5s");
    expect(formatDurationMs(3_000_000)).toBe("~50m");
  });

  it("formats hour-and-minute durations", () => {
    expect(formatDurationMs(3_600_000)).toBe("~1h");
    expect(formatDurationMs(7_500_000)).toBe("~2h 5m");
    expect(formatDurationMs(90_000_000)).toBe("~25h");
  });

  it("falls back to ~0s for negative or non-finite input", () => {
    expect(formatDurationMs(-1)).toBe("~0s");
    expect(formatDurationMs(Number.NaN)).toBe("~0s");
    expect(formatDurationMs(Number.POSITIVE_INFINITY)).toBe("~0s");
  });
});
