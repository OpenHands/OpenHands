import { describe, expect, it } from "vitest";
import {
  formatCompactTokenCount,
  getContextWindowUsagePercentage,
} from "#/utils/format-token-count";

describe("formatCompactTokenCount", () => {
  it.each([
    [500, "500"],
    [1_985, "2.0k"],
    [198_500, "198.5k"],
    [1_000_000, "1.0M"],
    [1_050_000, "1.1M"],
  ])("formats %i as %s", (value, expected) => {
    expect(formatCompactTokenCount(value)).toBe(expected);
  });
});

describe("getContextWindowUsagePercentage", () => {
  it("returns zero when the context window is unavailable", () => {
    expect(getContextWindowUsagePercentage(100, 0)).toBe(0);
  });

  it("caps usage at 100 percent", () => {
    expect(getContextWindowUsagePercentage(2_000_000, 1_000_000)).toBe(100);
  });

  it("computes the percentage from per-turn tokens", () => {
    expect(getContextWindowUsagePercentage(200_000, 1_000_000)).toBe(20);
  });
});
