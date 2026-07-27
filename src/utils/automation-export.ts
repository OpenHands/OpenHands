import type {
  Automation,
  AutomationExportFile,
  AutomationSpec,
  AutomationTrigger,
} from "#/types/automation";

const SUPPORTED_TRIGGER_TYPES = new Set(["cron", "schedule", "event"]);

type UnknownRecord = Record<string, unknown>;

export class AutomationFileValidationError extends Error {
  readonly issues: string[];

  constructor(issues: string[]) {
    super(issues.join("\n"));
    this.name = "AutomationFileValidationError";
    this.issues = issues;
  }
}

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validateRequiredString(
  record: UnknownRecord,
  field: string,
  path: string,
  issues: string[],
): string | undefined {
  const value = record[field];
  if (typeof value !== "string" || value.trim().length === 0) {
    issues.push(`${path}: expected a non-empty string`);
    return undefined;
  }
  return value;
}

function validateOptionalString(
  record: UnknownRecord,
  field: string,
  path: string,
  issues: string[],
): string | undefined {
  const value = record[field];
  if (value === undefined) return undefined;
  if (typeof value !== "string") {
    issues.push(`${path}: expected a string`);
    return undefined;
  }
  return value;
}

function validateTrigger(
  value: unknown,
  issues: string[],
): AutomationTrigger | undefined {
  if (!isRecord(value)) {
    issues.push("spec.trigger: expected an object");
    return undefined;
  }

  const type = validateRequiredString(
    value,
    "type",
    "spec.trigger.type",
    issues,
  );
  if (type && !SUPPORTED_TRIGGER_TYPES.has(type)) {
    issues.push(
      'spec.trigger.type: expected one of "cron", "schedule", or "event"',
    );
  }

  const schedule = validateOptionalString(
    value,
    "schedule",
    "spec.trigger.schedule",
    issues,
  );
  const scheduleHuman = validateOptionalString(
    value,
    "schedule_human",
    "spec.trigger.schedule_human",
    issues,
  );
  const timezone = validateOptionalString(
    value,
    "timezone",
    "spec.trigger.timezone",
    issues,
  );
  const source = validateOptionalString(
    value,
    "source",
    "spec.trigger.source",
    issues,
  );
  const filter = validateOptionalString(
    value,
    "filter",
    "spec.trigger.filter",
    issues,
  );

  let on: string | string[] | undefined;
  if (value.on !== undefined) {
    if (
      typeof value.on === "string" ||
      (Array.isArray(value.on) &&
        value.on.length > 0 &&
        value.on.every(
          (entry) => typeof entry === "string" && entry.trim().length > 0,
        ))
    ) {
      on = value.on as string | string[];
    } else {
      issues.push(
        "spec.trigger.on: expected a string or a non-empty array of strings",
      );
    }
  }

  if ((type === "cron" || type === "schedule") && !schedule?.trim()) {
    issues.push("spec.trigger.schedule: required for a scheduled trigger");
  }
  if (type === "event") {
    if (!source?.trim()) {
      issues.push("spec.trigger.source: required for an event trigger");
    }
    if (
      on === undefined ||
      (typeof on === "string" && on.trim().length === 0)
    ) {
      issues.push("spec.trigger.on: required for an event trigger");
    }
  }

  if (!type || !SUPPORTED_TRIGGER_TYPES.has(type)) return undefined;

  return {
    type,
    ...(schedule !== undefined && { schedule }),
    ...(scheduleHuman !== undefined && { schedule_human: scheduleHuman }),
    ...(timezone !== undefined && { timezone }),
    ...(source !== undefined && { source }),
    ...(on !== undefined && { on }),
    ...(filter !== undefined && { filter }),
  };
}

export function serializeAutomation(a: Automation): AutomationExportFile {
  const timezone = a.timezone ?? a.trigger.timezone;
  const spec: AutomationSpec = {
    name: a.name,
    trigger: { ...a.trigger },
    enabled: a.enabled,
    prompt: a.prompt,
    ...(a.repository !== undefined && { repository: a.repository }),
    ...(a.model !== undefined && { model: a.model }),
    ...(a.timeout != null && { timeout: a.timeout }),
    ...(a.branch !== undefined && { branch: a.branch }),
    ...(a.plugins !== undefined && { plugins: [...a.plugins] }),
    ...(a.notification !== undefined && { notification: a.notification }),
    ...(timezone !== undefined && { timezone }),
  };

  return { version: 1, kind: "automation", spec };
}

export function getAutomationExportFilename(
  automation: Pick<Automation, "id" | "name">,
): string {
  const slug = automation.name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return `${slug || automation.id}.automation.json`;
}

export function parseAutomationFile(json: unknown): AutomationSpec {
  const issues: string[] = [];
  if (!isRecord(json)) {
    throw new AutomationFileValidationError(["file: expected a JSON object"]);
  }

  if (json.version !== 1) {
    issues.push("version: expected 1");
  }
  if (json.kind !== "automation") {
    issues.push('kind: expected "automation"');
  }
  if (!isRecord(json.spec)) {
    issues.push("spec: expected an object");
    throw new AutomationFileValidationError(issues);
  }

  const name = validateRequiredString(json.spec, "name", "spec.name", issues);
  const prompt = validateRequiredString(
    json.spec,
    "prompt",
    "spec.prompt",
    issues,
  );
  const trigger = validateTrigger(json.spec.trigger, issues);

  const enabled = json.spec.enabled;
  if (typeof enabled !== "boolean") {
    issues.push("spec.enabled: expected a boolean");
  }

  const repository = validateOptionalString(
    json.spec,
    "repository",
    "spec.repository",
    issues,
  );
  const branch = validateOptionalString(
    json.spec,
    "branch",
    "spec.branch",
    issues,
  );
  const notification = validateOptionalString(
    json.spec,
    "notification",
    "spec.notification",
    issues,
  );
  const timezone = validateOptionalString(
    json.spec,
    "timezone",
    "spec.timezone",
    issues,
  );

  let model: string | null | undefined;
  if (json.spec.model === null || json.spec.model === undefined) {
    model = json.spec.model;
  } else if (
    typeof json.spec.model === "string" &&
    json.spec.model.trim().length > 0
  ) {
    model = json.spec.model;
  } else {
    issues.push("spec.model: expected a non-empty string or null");
  }

  let timeout: number | null | undefined;
  if (json.spec.timeout === null || json.spec.timeout === undefined) {
    timeout = json.spec.timeout;
  } else if (
    typeof json.spec.timeout === "number" &&
    Number.isInteger(json.spec.timeout) &&
    json.spec.timeout > 0
  ) {
    timeout = json.spec.timeout;
  } else {
    issues.push("spec.timeout: expected a positive integer");
  }

  let plugins: string[] | undefined;
  if (json.spec.plugins !== undefined) {
    if (
      Array.isArray(json.spec.plugins) &&
      json.spec.plugins.every(
        (plugin) => typeof plugin === "string" && plugin.trim().length > 0,
      )
    ) {
      plugins = [...json.spec.plugins] as string[];
    } else {
      issues.push("spec.plugins: expected an array of non-empty strings");
    }
  }

  if (issues.length > 0 || !name || !prompt || !trigger) {
    throw new AutomationFileValidationError(issues);
  }

  return {
    name,
    prompt,
    trigger,
    enabled: enabled as boolean,
    ...(repository !== undefined && { repository }),
    ...(model !== undefined && { model }),
    ...(timeout !== undefined && { timeout }),
    ...(branch !== undefined && { branch }),
    ...(plugins !== undefined && { plugins }),
    ...(notification !== undefined && { notification }),
    ...(timezone !== undefined && { timezone }),
  };
}
