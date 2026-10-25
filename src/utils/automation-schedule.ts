import { I18nKey } from "#/i18n/declaration";

export type SchedulePresetKind = "daily" | "weekdays" | "weekly";

export interface PresetSchedule {
  kind: SchedulePresetKind;
  hour: number;
  minute: number;
  weekday?: number;
}

export interface CustomSchedule {
  kind: "custom";
  raw: string;
  hour?: number;
  minute?: number;
}

export type ParsedSchedule = PresetSchedule | CustomSchedule;

const SINGLE_INT = /^(\d+)$/;

function parseSingleInt(
  field: string,
  min: number,
  max: number,
): number | null {
  const match = field.match(SINGLE_INT);
  if (!match) return null;
  const value = Number(match[1]);
  if (Number.isNaN(value) || value < min || value > max) return null;
  return value;
}

export function parseCronSchedule(
  cron: string | undefined | null,
): ParsedSchedule {
  const raw = (cron ?? "").trim();
  if (!raw) return { kind: "custom", raw: "" };

  const fields = raw.split(/\s+/);
  if (fields.length !== 5) return { kind: "custom", raw };

  const [minuteField, hourField, domField, monthField, dowField] = fields;

  const minute = parseSingleInt(minuteField, 0, 59);
  const hour = parseSingleInt(hourField, 0, 23);

  if (minute === null || hour === null) {
    return { kind: "custom", raw };
  }
  if (domField !== "*" || monthField !== "*") {
    return { kind: "custom", raw, hour, minute };
  }

  if (dowField === "*" || dowField === "0-6") {
    return { kind: "daily", hour, minute };
  }
  if (dowField === "1-5") {
    return { kind: "weekdays", hour, minute };
  }
  const weekday = parseSingleInt(dowField, 0, 6);
  if (weekday !== null) {
    return { kind: "weekly", hour, minute, weekday };
  }
  return { kind: "custom", raw, hour, minute };
}

export function formatTimeOfDay(hour: number, minute: number): string {
  const hh = String(hour).padStart(2, "0");
  const mm = String(minute).padStart(2, "0");
  return `${hh}:${mm}`;
}

export function formatEventOn(on: string | string[] | undefined): string {
  if (!on) return "—";
  if (Array.isArray(on)) return on.join(", ");
  return on;
}

export type CronScheduleValidation =
  | { schedule: string }
  | { errorKey: I18nKey };
