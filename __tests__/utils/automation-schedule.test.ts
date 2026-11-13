import { describe, it, expect } from "vitest";
import { parseCronSchedule } from "#/utils/automation-schedule";

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
});
