import { describe, expect, it, vi, beforeEach } from "vitest";
import { formatTimeDelta, parseDateAsUTC } from "#/utils/format-time-delta";

describe("parseDateAsUTC", () => {
  it("preserves explicit timezone indicators (Z, z, +00:00, -03:00, +02:00, +0000, -0300)", () => {
    const expectedEpoch = Date.parse("2024-01-01T00:00:00Z");

    expect(parseDateAsUTC("2024-01-01T00:00:00Z").getTime()).toBe(expectedEpoch);
    expect(parseDateAsUTC("2024-01-01T00:00:00z").getTime()).toBe(expectedEpoch);
    expect(parseDateAsUTC("2024-01-01T00:00:00+00:00").getTime()).toBe(
      expectedEpoch,
    );
    expect(parseDateAsUTC("2023-12-31T21:00:00-03:00").getTime()).toBe(
      expectedEpoch,
    );
    expect(parseDateAsUTC("2024-01-01T02:00:00+02:00").getTime()).toBe(
      expectedEpoch,
    );
    expect(parseDateAsUTC("2023-12-31T21:00:00-0300").getTime()).toBe(
      expectedEpoch,
    );
    expect(parseDateAsUTC("2024-01-01T00:00:00+0000").getTime()).toBe(
      expectedEpoch,
    );
  });

  it("appends Z to suffix-less date strings to force UTC interpretation (legacy fallback)", () => {
    expect(parseDateAsUTC("2024-01-01T00:00:00").getTime()).toBe(
      Date.parse("2024-01-01T00:00:00Z"),
    );
    expect(parseDateAsUTC("2023-12-31T21:00:00").getTime()).toBe(
      Date.parse("2023-12-31T21:00:00Z"),
    );
  });
});

describe("formatTimeDelta", () => {
  beforeEach(() => {
    const now = new Date("2024-01-01T00:00:00Z");
    vi.useFakeTimers({ now });
  });

  it("formats offset-aware strings from local and remote agent servers consistently", () => {
    // Current time is 2024-01-01T00:00:00Z (21:00 local in UTC-3)
    // Server emitting UTC offset or local host offset for an event sent 10s ago:
    expect(formatTimeDelta("2023-12-31T23:59:50+00:00")).toBe("10s");
    expect(formatTimeDelta("2023-12-31T20:59:50-03:00")).toBe("10s");
    expect(formatTimeDelta("2024-01-01T01:59:50+02:00")).toBe("10s");

    // Message sent 5m ago with explicit offset:
    expect(formatTimeDelta("2023-12-31T20:55:00-03:00")).toBe("5m");
    expect(formatTimeDelta("2023-12-31T23:55:00+00:00")).toBe("5m");

    // Legacy suffix-less string without offset falls back to UTC:
    // 21:00 local without offset is read as 21:00 UTC, which is 3 hours before 00:00 UTC
    expect(formatTimeDelta("2023-12-31T21:00:00")).toBe("3h");
  });

  it("formats the yearly time correctly", () => {
    const oneYearAgo = new Date("2023-01-01T00:00:00Z");
    expect(formatTimeDelta(oneYearAgo)).toBe("1y");

    const twoYearsAgo = new Date("2022-01-01T00:00:00Z");
    expect(formatTimeDelta(twoYearsAgo)).toBe("2y");

    const threeYearsAgo = new Date("2021-01-01T00:00:00Z");
    expect(formatTimeDelta(threeYearsAgo)).toBe("3y");
  });

  it("formats the monthly time correctly", () => {
    const oneMonthAgo = new Date("2023-12-01T00:00:00Z");
    expect(formatTimeDelta(oneMonthAgo)).toBe("1mo");

    const twoMonthsAgo = new Date("2023-11-01T00:00:00Z");
    expect(formatTimeDelta(twoMonthsAgo)).toBe("2mo");

    const threeMonthsAgo = new Date("2023-10-01T00:00:00Z");
    expect(formatTimeDelta(threeMonthsAgo)).toBe("3mo");
  });

  it("formats the daily time correctly", () => {
    const oneDayAgo = new Date("2023-12-31T00:00:00Z");
    expect(formatTimeDelta(oneDayAgo)).toBe("1d");

    const twoDaysAgo = new Date("2023-12-30T00:00:00Z");
    expect(formatTimeDelta(twoDaysAgo)).toBe("2d");

    const threeDaysAgo = new Date("2023-12-29T00:00:00Z");
    expect(formatTimeDelta(threeDaysAgo)).toBe("3d");
  });

  it("formats the hourly time correctly", () => {
    const oneHourAgo = new Date("2023-12-31T23:00:00Z");
    expect(formatTimeDelta(oneHourAgo)).toBe("1h");

    const twoHoursAgo = new Date("2023-12-31T22:00:00Z");
    expect(formatTimeDelta(twoHoursAgo)).toBe("2h");

    const threeHoursAgo = new Date("2023-12-31T21:00:00Z");
    expect(formatTimeDelta(threeHoursAgo)).toBe("3h");
  });

  it("formats the minute time correctly", () => {
    const oneMinuteAgo = new Date("2023-12-31T23:59:00Z");
    expect(formatTimeDelta(oneMinuteAgo)).toBe("1m");

    const twoMinutesAgo = new Date("2023-12-31T23:58:00Z");
    expect(formatTimeDelta(twoMinutesAgo)).toBe("2m");

    const threeMinutesAgo = new Date("2023-12-31T23:57:00Z");
    expect(formatTimeDelta(threeMinutesAgo)).toBe("3m");
  });

  it("formats the second time correctly", () => {
    const oneSecondAgo = new Date("2023-12-31T23:59:59Z");
    expect(formatTimeDelta(oneSecondAgo)).toBe("1s");

    const twoSecondsAgo = new Date("2023-12-31T23:59:58Z");
    expect(formatTimeDelta(twoSecondsAgo)).toBe("2s");

    const threeSecondsAgo = new Date("2023-12-31T23:59:57Z");
    expect(formatTimeDelta(threeSecondsAgo)).toBe("3s");
  });
});
