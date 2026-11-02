import { describe, expect, it } from "vitest";
import {
  formatEventTimestamp,
  formatRelativeEventTimestamp,
} from "#/utils/format-event-timestamp";

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

describe("formatRelativeEventTimestamp", () => {
  const now = new Date(2026, 10, 2, 12).getTime();
  it.each([
    [0, "Just now"],
    [59, "Just now"],
    [60, "1 minute ago"],
    [1800, "30 minutes ago"],
    [3600, "1 hour ago"],
    [18000, "5 hours ago"],
    [86399, "23 hours ago"],
    [86400, "1 day ago"],
    [604799, "6 days ago"],
    [604800, "Oct 26"],
  ])("formats a message %s seconds old", (seconds, expected) => {
    expect(
      formatRelativeEventTimestamp(
        new Date(now - Number(seconds) * 1000).toISOString(),
        "en",
        "Just now",
        now,
      ),
    ).toBe(expected);
  });

  it("localizes relative labels and includes the year for older dates", () => {
    expect(
      formatRelativeEventTimestamp(
        new Date(now - 1800000).toISOString(),
        "fr",
        "À l'instant",
        now,
      ),
    ).toBe("il y a 30 minutes");
    expect(
      formatRelativeEventTimestamp(
        new Date(2025, 9, 22, 12).toISOString(),
        "en",
        "Just now",
        now,
      ),
    ).toBe("Oct 22, 2025");
  });

  it("uses UTC for offset-less timestamps and avoids invalid or future ages", () => {
    const utcNow = Date.parse("2026-11-02T12:30:00Z");
    expect(
      formatRelativeEventTimestamp(
        "2026-11-02T12:00:00",
        "en",
        "Just now",
        utcNow,
      ),
    ).toBe("30 minutes ago");
    expect(
      formatRelativeEventTimestamp(
        "2026-11-02T14:00:00+02:00",
        "en",
        "Just now",
        utcNow,
      ),
    ).toBe("30 minutes ago");
    for (const value of [undefined, "not-a-date"])
      expect(
        formatRelativeEventTimestamp(value, "en", "Just now", now),
      ).toBeNull();
    expect(
      formatRelativeEventTimestamp(
        new Date(now + 1000).toISOString(),
        "en",
        "Just now",
        now,
      ),
    ).toBe("Just now");
  });
});
