import { afterEach, describe, expect, it, vi } from "vitest";
import { I18nKey } from "#/i18n/declaration";
import { formatRelativeTime } from "#/utils/format-relative-time";

const t = (key: string, options?: Record<string, unknown>) =>
  options?.count === undefined ? key : `${key}:${options.count}`;

describe("formatRelativeTime", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it.each([
    {
      description: "less than a minute ago",
      lastRunAt: "2026-06-15T11:59:30Z",
      expected: I18nKey.AUTOMATIONS$DETAIL$TIME_JUST_NOW,
    },
    {
      description: "minutes ago",
      lastRunAt: "2026-06-15T11:23:00Z",
      expected: `${I18nKey.AUTOMATIONS$DETAIL$TIME_MINUTES_AGO}:37`,
    },
    {
      description: "exactly one minute ago",
      lastRunAt: "2026-06-15T11:59:00Z",
      expected: `${I18nKey.AUTOMATIONS$DETAIL$TIME_MINUTES_AGO}:1`,
    },
    {
      description: "hours ago",
      lastRunAt: "2026-06-15T06:30:00Z",
      expected: `${I18nKey.AUTOMATIONS$DETAIL$TIME_HOURS_AGO}:5`,
    },
    {
      description: "exactly one hour ago",
      lastRunAt: "2026-06-15T11:00:00Z",
      expected: `${I18nKey.AUTOMATIONS$DETAIL$TIME_HOURS_AGO}:1`,
    },
    {
      description: "yesterday",
      lastRunAt: "2026-06-14T10:00:00Z",
      expected: I18nKey.AUTOMATIONS$DETAIL$TIME_YESTERDAY,
    },
    {
      description: "exactly one day ago",
      lastRunAt: "2026-06-14T12:00:00Z",
      expected: I18nKey.AUTOMATIONS$DETAIL$TIME_YESTERDAY,
    },
    {
      description: "several days ago",
      lastRunAt: "2026-06-12T07:00:00Z",
      expected: `${I18nKey.AUTOMATIONS$DETAIL$TIME_DAYS_AGO}:3`,
    },
    {
      description: "exactly one week ago",
      lastRunAt: "2026-06-08T12:00:00Z",
      expected: "Jun 8, 2026",
    },
  ])("words a run from $description", ({ lastRunAt, expected }) => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-15T12:00:00Z"));

    expect(formatRelativeTime(lastRunAt, "en-US", t)).toBe(expected);
  });
});
