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

export function buildCronSchedule(input: PresetSchedule): string {
  const { kind, hour, minute, weekday } = input;
  switch (kind) {
    case "daily":
      return `${minute} ${hour} * * *`;
    case "weekdays":
      return `${minute} ${hour} * * 1-5`;
    case "weekly":
      return `${minute} ${hour} * * ${weekday ?? 1}`;
    default: {
      const _exhaustive: never = kind;
      return _exhaustive;
    }
  }
}

export function formatTimeOfDay(hour: number, minute: number): string {
  const hh = String(hour).padStart(2, "0");
  const mm = String(minute).padStart(2, "0");
  return `${hh}:${mm}`;
}

export function parseTimeOfDay(
  value: string,
): { hour: number; minute: number } | null {
  const match = value.match(/^(\d{1,2}):(\d{2})$/);
  if (!match) return null;
  const hour = Number(match[1]);
  const minute = Number(match[2]);
  if (
    Number.isNaN(hour) ||
    Number.isNaN(minute) ||
    hour < 0 ||
    hour > 23 ||
    minute < 0 ||
    minute > 59
  ) {
    return null;
  }
  return { hour, minute };
}

export function formatEventOn(on: string | string[] | undefined): string {
  if (!on) return "—";
  if (Array.isArray(on)) return on.join(", ");
  return on;
}

/** Example expression shown when the cron field is empty. */
export const CRON_EXPRESSION_EXAMPLE = "*/10 * * * *";

// The service validates with croniter, so this must accept everything croniter
// does: five fields, optionally plus seconds and then year, or an alias.
const CRON_ALIASES = new Set([
  "@yearly",
  "@annually",
  "@monthly",
  "@weekly",
  "@daily",
  "@midnight",
  "@hourly",
]);

const CRON_MIN_FIELDS = 5;
const CRON_MAX_FIELDS = 7;

// Seconds and year are appended after these, so the positions always hold.
const CRON_FIELD_BOUNDS: readonly (readonly [number, number])[] = [
  [0, 59], // minute
  [0, 23], // hour
  [1, 31], // day of month
  [1, 12], // month
  [0, 7], // day of week — croniter spells Sunday 0 and 7
];

const DAY_OF_MONTH_INDEX = 2;
const MONTH_INDEX = 3;

// Indexed from 1. February is 29: the 29th does fire, in leap years.
const MAX_DAYS_IN_MONTH = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

const MONTH_NAMES = "JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC";
const DAY_NAMES = "SUN|MON|TUE|WED|THU|FRI|SAT";

// croniter does not bound the step: `*/90` is accepted.
const NUMERIC_TERM = /^(\*|\d+(?:-\d+)?)(?:\/(\d+))?$/;

/** Vocabulary beyond plain numbers, matched but not expanded. */
const EXTRA_TERM_PATTERNS: readonly (RegExp | null)[] = [
  null, // minute
  null, // hour
  /^(?:\?|L|\d+W)$/i, // `?`, `L`, `15W`
  new RegExp(`^(?:${MONTH_NAMES})(?:-(?:${MONTH_NAMES}))?(?:/\\d+)?$`, "i"),
  // `SUN`, `MON-FRI#2`, `5#3`, `?` — but no `L`, which croniter rejects here.
  new RegExp(
    `^(?:(?:${DAY_NAMES}|\\d+)(?:-(?:${DAY_NAMES}|\\d+))?(?:/\\d+)?(?:#\\d+)?|\\?)$`,
    "i",
  ),
];

type CronFieldParse =
  | { kind: "invalid" }
  /** Well-formed, but not expanded, so it constrains nothing. */
  | { kind: "unmodelled" }
  | { kind: "values"; values: number[] };

function parseCronField(
  field: string,
  [min, max]: readonly [number, number],
  extraPattern: RegExp | null,
): CronFieldParse {
  const values = new Set<number>();

  for (const term of field.split(",")) {
    const match = term.match(NUMERIC_TERM);
    if (!match) {
      if (extraPattern?.test(term)) return { kind: "unmodelled" };
      return { kind: "invalid" };
    }

    const [, rangePart, stepPart] = match;
    const step = stepPart === undefined ? 1 : Number(stepPart);
    if (step < 1) return { kind: "invalid" };

    if (rangePart === "*") {
      for (let value = min; value <= max; value += step) values.add(value);
      continue;
    }

    const [startPart, endPart] = rangePart.split("-");
    const start = parseSingleInt(startPart, min, max);
    if (start === null) return { kind: "invalid" };
    if (endPart === undefined) {
      values.add(start);
      continue;
    }
    const end = parseSingleInt(endPart, min, max);
    if (end === null) return { kind: "invalid" };
    // croniter wraps a reversed range like `5-1`; that wrap is not modelled.
    if (end < start) return { kind: "unmodelled" };
    for (let value = start; value <= end; value += step) values.add(value);
  }

  return { kind: "values", values: [...values] };
}

export type CronScheduleValidation =
  | { schedule: string }
  | { errorKey: I18nKey };

/**
 * Validate a raw cron expression from the edit form — `parseCronSchedule` only
 * classifies against the presets, reporting arbitrary text as `custom`.
 *
 * One-sided by design: rejects only what croniter certainly rejects, and defers
 * on anything it does not model.
 */
export function validateCronSchedule(raw: string): CronScheduleValidation {
  const schedule = raw.trim();
  if (!schedule) return { errorKey: I18nKey.AUTOMATIONS$ERROR_CRON_INVALID };

  if (schedule.startsWith("@")) {
    return CRON_ALIASES.has(schedule.toLowerCase())
      ? { schedule }
      : { errorKey: I18nKey.AUTOMATIONS$ERROR_CRON_INVALID };
  }

  const fields = schedule.split(/\s+/);
  if (fields.length < CRON_MIN_FIELDS || fields.length > CRON_MAX_FIELDS) {
    return { errorKey: I18nKey.AUTOMATIONS$ERROR_CRON_INVALID };
  }

  const parsed = CRON_FIELD_BOUNDS.map((bounds, index) =>
    parseCronField(fields[index], bounds, EXTRA_TERM_PATTERNS[index]),
  );
  if (parsed.some((field) => field.kind === "invalid")) {
    return { errorKey: I18nKey.AUTOMATIONS$ERROR_CRON_INVALID };
  }

  // croniter parses `0 0 31 2 *` and then finds no fire time for it.
  const days = parsed[DAY_OF_MONTH_INDEX];
  const months = parsed[MONTH_INDEX];
  if (
    days.kind === "values" &&
    months.kind === "values" &&
    !months.values.some((month) =>
      days.values.some((day) => day <= MAX_DAYS_IN_MONTH[month]),
    )
  ) {
    return { errorKey: I18nKey.AUTOMATIONS$ERROR_CRON_UNREACHABLE };
  }

  return { schedule };
}
