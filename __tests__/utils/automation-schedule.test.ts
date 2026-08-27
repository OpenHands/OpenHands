import { describe, it, expect } from "vitest";
import { I18nKey } from "#/i18n/declaration";
import {
  buildCronSchedule,
  validateCronSchedule,
  parseCronSchedule,
  parseTimeOfDay,
} from "#/utils/automation-schedule";

describe("automation-schedule", () => {
  describe("parseCronSchedule", () => {
    it("decodes Daily / Weekdays / Weekly preset cron expressions", () => {
      // Arrange — three representative preset strings the edit modal
      // needs to round-trip into UI state.
      const cases = {
        daily: "0 9 * * *",
        weekdays: "30 8 * * 1-5",
        weekly: "0 14 * * 3",
      };

      // Act
      const daily = parseCronSchedule(cases.daily);
      const weekdays = parseCronSchedule(cases.weekdays);
      const weekly = parseCronSchedule(cases.weekly);

      // Assert
      expect(daily).toEqual({ kind: "daily", hour: 9, minute: 0 });
      expect(weekdays).toEqual({ kind: "weekdays", hour: 8, minute: 30 });
      expect(weekly).toEqual({
        kind: "weekly",
        hour: 14,
        minute: 0,
        weekday: 3,
      });
    });

    it("falls back to 'custom' for cron strings that don't match a preset", () => {
      // Arrange — schedules the UI must NOT silently rewrite when saving:
      // multi-value hour, monthly DOM, missing fields, garbage.
      const inputs = ["0 9,17 * * *", "0 9 1 * *", "every 5 minutes", ""];

      // Act
      const results = inputs.map(parseCronSchedule);

      // Assert — every non-preset stays as kind: "custom".
      expect(results.every((r) => r.kind === "custom")).toBe(true);
    });
  });

  describe("buildCronSchedule", () => {
    it("emits canonical cron strings for each preset kind", () => {
      // Act
      const daily = buildCronSchedule({ kind: "daily", hour: 9, minute: 0 });
      const weekdays = buildCronSchedule({
        kind: "weekdays",
        hour: 8,
        minute: 30,
      });
      const weekly = buildCronSchedule({
        kind: "weekly",
        hour: 14,
        minute: 0,
        weekday: 3,
      });

      // Assert
      expect(daily).toBe("0 9 * * *");
      expect(weekdays).toBe("30 8 * * 1-5");
      expect(weekly).toBe("0 14 * * 3");
    });
  });

  describe("validateCronSchedule", () => {
    it("accepts, and trims, expressions the preset parser reports as custom", () => {
      // Arrange — none of these map onto Daily/Weekdays/Weekly, so the
      // edit modal relies on this check rather than parseCronSchedule.
      const valid = ["*/10 * * * *", "0 9,17 * * *", "0 0 1-15 * 1-5"];

      // Assert
      expect(valid.map((c) => validateCronSchedule(c))).toEqual(
        valid.map((schedule) => ({ schedule })),
      );
      expect(valid.every((c) => parseCronSchedule(c).kind === "custom")).toBe(
        true,
      );
      expect(validateCronSchedule("  */10 * * * *  ")).toEqual({
        schedule: "*/10 * * * *",
      });
    });

    it("accepts the expressions the automation service accepts", () => {
      // Arrange — verified against croniter 6.2.2, the validator behind the
      // service's CronTrigger.schedule. Rejecting any of these here would
      // lock an existing automation out of the edit form entirely.
      const serviceAccepts = [
        "0 0 * * 7", // croniter spells Sunday as both 0 and 7
        "0 0 * * SUN", // day names
        "0 0 * JAN *", // month names
        "0 0 * * MON-FRI", // a range of day names
        "0 0 L * *", // last day of the month
        "0 0 15W * *", // nearest weekday to the 15th
        "0 0 * * 5#3", // third Friday
        "0 0 ? * MON", // `?` in the day-of-month field
        "5-1 * * * *", // a reversed range wraps around
        "*/90 * * * *", // croniter does not bound the step
        "* * * * * *", // trailing seconds field
        "* * * * * * *", // trailing seconds and year fields
        "@daily", // alias
        "0 0 29 2 *", // fires only in leap years, but it does fire
        "0 0 31 1,2 *", // January has a 31st even though February does not
      ];

      // Assert
      expect(serviceAccepts.map((c) => validateCronSchedule(c))).toEqual(
        serviceAccepts.map((schedule) => ({ schedule })),
      );
    });

    it("rejects wrong field counts, out-of-range values and free text", () => {
      // Arrange — croniter rejects each of these too.
      const invalid = [
        "* * * *",
        "* * * * * * * *",
        "60 * * * *",
        "* * * * 9",
        "every ten minutes please now",
        "@bogus",
        "",
      ];

      // Assert — one shared key, so the form shows the same guidance.
      expect(invalid.map((c) => validateCronSchedule(c))).toEqual(
        invalid.map(() => ({
          errorKey: I18nKey.AUTOMATIONS$ERROR_CRON_INVALID,
        })),
      );
    });

    it("rejects a well-formed schedule that can never fire", () => {
      // Arrange — croniter parses these and then fails to find a fire time,
      // so the service rejects them. Catching them here keeps the request
      // from being sent at all.
      const unreachable = [
        "0 0 31 2 *",
        "0 0 30 2 *",
        "0 0 31 4 *",
        "0 0 31 2,4 *",
        "0 0 31 2 MON",
      ];

      // Assert — a distinct key: the expression is well-formed, not malformed.
      expect(unreachable.map((c) => validateCronSchedule(c))).toEqual(
        unreachable.map(() => ({
          errorKey: I18nKey.AUTOMATIONS$ERROR_CRON_UNREACHABLE,
        })),
      );
    });
  });

  describe("parseTimeOfDay", () => {
    it("parses HH:MM and rejects out-of-range or malformed values", () => {
      // Act
      const valid = parseTimeOfDay("09:30");
      const invalidHour = parseTimeOfDay("24:00");
      const malformed = parseTimeOfDay("9-30");

      // Assert
      expect(valid).toEqual({ hour: 9, minute: 30 });
      expect(invalidHour).toBeNull();
      expect(malformed).toBeNull();
    });
  });
});
