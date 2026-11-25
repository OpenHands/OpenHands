import { describe, expect, it } from "vitest";
import { formatClock } from "#/utils/format-clock";

describe("formatClock", () => {
  it("formats zero as 00:00", () => {
    expect(formatClock(0)).toBe("00:00");
  });

  it("pads single-digit minutes and seconds", () => {
    expect(formatClock(5)).toBe("00:05");
    expect(formatClock(59)).toBe("00:59");
  });

  it("rolls over into minutes", () => {
    expect(formatClock(60)).toBe("01:00");
    expect(formatClock(65)).toBe("01:05");
    expect(formatClock(600)).toBe("10:00");
  });

  it("keeps counting minutes past 59 instead of wrapping to hours", () => {
    expect(formatClock(3661)).toBe("61:01");
  });

  it("clamps negative and fractional input to elapsed whole seconds", () => {
    expect(formatClock(-5)).toBe("00:00");
    expect(formatClock(12.9)).toBe("00:12");
  });
});
