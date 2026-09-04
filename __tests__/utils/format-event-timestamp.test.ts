import { describe, expect, it } from "vitest";
import { formatEventTimestamp } from "#/utils/format-event-timestamp";

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
