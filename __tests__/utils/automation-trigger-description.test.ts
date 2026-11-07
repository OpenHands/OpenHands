import { describe, expect, it } from "vitest";
import type { TFunction } from "i18next";
import translations from "#/i18n/translation.json";
import type { AutomationTrigger } from "#/types/automation";
import { describeSchedule } from "#/utils/automation-trigger-description";

// English copy with interpolation, so assertions read like the UI.
const t = ((key: string, options?: Record<string, unknown>) => {
  const entry = (translations as Record<string, { en: string }>)[key];
  return entry.en.replace(/{{(\w+)}}/g, (_, name) => String(options?.[name]));
}) as TFunction;

const cron = (schedule: string, extra: Partial<AutomationTrigger> = {}) => ({
  type: "cron",
  schedule,
  ...extra,
});

describe("describeSchedule", () => {
  it.each([
    { schedule: "0 9 * * *", expected: "Every day at 9:00 AM" },
    { schedule: "30 17 * * 1-5", expected: "Weekdays at 5:30 PM" },
    { schedule: "0 8 * * 1", expected: "Every Monday at 8:00 AM" },
    { schedule: "* * * * *", expected: "Every minute" },
    { schedule: "*/15 * * * *", expected: "Every 15 minutes" },
    { schedule: "5 * * * *", expected: "Every hour at :05" },
  ])("words $schedule as “$expected”", ({ schedule, expected }) => {
    expect(describeSchedule(cron(schedule), t, "en-US")).toBe(expected);
  });

  it("falls back to the service's wording, then the raw cron", () => {
    const uncommon = "0 9 1 * *";

    expect(
      describeSchedule(
        cron(uncommon, { schedule_human: "Monthly on the 1st" }),
        t,
        "en-US",
      ),
    ).toBe("Monthly on the 1st");
    expect(describeSchedule(cron(uncommon), t, "en-US")).toBe(uncommon);
  });

  it("has nothing to say about an event trigger", () => {
    expect(describeSchedule({ type: "event" }, t, "en-US")).toBeNull();
  });
});
