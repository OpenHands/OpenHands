import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  formatCompactTokenCount,
  getContextWindowUsagePercentage,
} from "#/utils/format-token-count";
import { formatEventTimestamp } from "#/utils/format-event-timestamp";
import { formatTimeDelta } from "#/utils/format-time-delta";
import { getAgentStateEmoji } from "#/utils/agent-state-emoji";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

describe("format-token-count", () => {
  describe("formatCompactTokenCount", () => {
    it.each([
      [500, "500"],
      [1_985, "2.0k"],
      [198_500, "198.5k"],
      [999_949, "999.9k"],
      [1_000_000, "1.0M"],
      [1_050_000, "1.1M"],
    ])("formats %i as %s", (value, expected) => {
      expect(formatCompactTokenCount(value)).toBe(expected);
    });

    it("promotes to the next unit when rounding reaches it", () => {
      expect(formatCompactTokenCount(999_999)).toBe("1.0M");
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
});

describe("format-event-timestamp", () => {
  describe("formatEventTimestamp", () => {
    it.each(["en", "zh-CN"])("uses the supplied %s locale", (locale) => {
      const timestamp = "2026-04-16T19:32:29.828Z";

      expect(formatEventTimestamp(timestamp, locale)).toBe(
        new Date(timestamp).toLocaleString(locale, {
          dateStyle: "medium",
          timeStyle: "short",
        }),
      );
    });
    it("reads an offset-less agent-server timestamp as UTC", () => {
      expect(formatEventTimestamp("2026-04-16T19:32:29.828069", "en")).toBe(
        formatEventTimestamp("2026-04-16T19:32:29.828069Z", "en"),
      );
    });
  });
});

describe("format-time-delta", () => {
  describe("formatTimeDelta", () => {
    beforeEach(() => {
      const now = new Date("2024-01-01T00:00:00Z");
      vi.useFakeTimers({ now });
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
});

describe("agent-state-emoji", () => {
  describe("getAgentStateEmoji", () => {
    it.each([
      [ExecutionStatus.RUNNING, "🟢"],
      [ExecutionStatus.FINISHED, "✅"],
      [ExecutionStatus.IDLE, "✅"],
      [ExecutionStatus.WAITING_FOR_CONFIRMATION, "✅"],
      [ExecutionStatus.PAUSED, "⚪"],
      [ExecutionStatus.ERROR, "🔴"],
      [ExecutionStatus.STUCK, "🔴"],
    ])("returns %s for execution status %s", (status, emoji) => {
      expect(getAgentStateEmoji(status)).toBe(emoji);
    });

    it("returns null for an unknown / missing status", () => {
      expect(getAgentStateEmoji(null)).toBeNull();
      expect(getAgentStateEmoji(undefined)).toBeNull();
    });
  });
});
