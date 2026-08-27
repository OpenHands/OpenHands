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

// The automation service validates with croniter, so this form must accept
// every expression croniter does — otherwise a schedule the service is happy
// to store cannot be saved from the UI. croniter takes five fields, optionally
// followed by a seconds field and then a year field, or one of these aliases.
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

// Bounds of the five positional fields, in field order. The optional seconds
// and year fields are appended *after* these, so these positions hold for
// every field count croniter accepts.
const CRON_FIELD_BOUNDS: readonly (readonly [number, number])[] = [
  [0, 59], // minute
  [0, 23], // hour
  [1, 31], // day of month
  [1, 12], // month
  [0, 7], // day of week — croniter spells Sunday as both 0 and 7
];

const DAY_OF_MONTH_INDEX = 2;
const MONTH_INDEX = 3;

// Longest each month can be, indexed from 1. February is 29 because a schedule
// on the 29th does fire, in leap years.
const MAX_DAYS_IN_MONTH = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

const MONTH_NAMES = "JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC";
const DAY_NAMES = "SUN|MON|TUE|WED|THU|FRI|SAT";

// A term this form models numerically: `*`, `5` or `1-5`, each optionally with
// a `/step`. croniter does not bound the step (`*/90` is accepted), so neither
// does this.
const NUMERIC_TERM = /^(\*|\d+(?:-\d+)?)(?:\/(\d+))?$/;

/**
 * Per-field vocabulary beyond plain numbers. croniter takes month and day
 * names, and `L`/`W`/`#`/`?` in the two day fields. Terms matching these are
 * accepted without being expanded — enough to know the expression is
 * well-formed, and the service is left to judge the rest.
 */
const EXTRA_TERM_PATTERNS: readonly (RegExp | null)[] = [
  null, // minute — numbers only
  null, // hour — numbers only
  /^(?:\?|L|\d+W)$/i, // day of month — `?`, `L`, `15W`
  // month — `JAN`, `JAN-MAR`, `JAN-DEC/3`
  new RegExp(`^(?:${MONTH_NAMES})(?:-(?:${MONTH_NAMES}))?(?:/\\d+)?$`, "i"),
  // day of week — `SUN`, `MON-FRI`, `SUN/2`, `MON-FRI#2`, `5#3`, `?`. croniter
  // takes no `L` here, so `5L` falls through to the numeric check and is
  // rejected, as croniter rejects it.
  new RegExp(
    `^(?:(?:${DAY_NAMES}|\\d+)(?:-(?:${DAY_NAMES}|\\d+))?(?:/\\d+)?(?:#\\d+)?|\\?)$`,
    "i",
  ),
];

type CronFieldParse =
  /** Rejected outright: croniter would reject it too. */
  | { kind: "invalid" }
  /** Well-formed but not expanded, so it constrains nothing below. */
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
    // croniter accepts a reversed range such as `5-1`, which wraps around.
    // That wrap is not modelled here, so the field constrains nothing.
    if (end < start) return { kind: "unmodelled" };
    for (let value = start; value <= end; value += step) values.add(value);
  }

  return { kind: "values", values: [...values] };
}

export type CronScheduleValidation =
  | { schedule: string }
  | { errorKey: I18nKey };

/**
 * Validate a raw cron expression from the edit form. `parseCronSchedule` only
 * classifies an expression against the UI presets, so it reports arbitrary
 * text as `custom` rather than rejecting it.
 *
 * The service is authoritative, so this check is deliberately one-sided: it
 * rejects only what croniter certainly rejects, and defers on any construct it
 * does not model. The one semantic check it does make is the one croniter
 * itself makes after parsing — that the schedule can fire at all.
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

  // croniter accepts `0 0 31 2 *` as well-formed and then fails to find a fire
  // time for it. Reject only when no day the expression allows can occur in
  // any month it allows, so an expression that merely fires rarely survives.
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
