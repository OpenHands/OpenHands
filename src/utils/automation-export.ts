import { getImportExportSpec } from "#/manifests/automation-interface";
import type {
  Automation,
  AutomationExportFile,
  AutomationScriptBundle,
  AutomationSpec,
  AutomationTrigger,
} from "#/types/automation";
import { base64ToBytes, bytesToBase64 } from "#/utils/base64";

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

function decodesAsBase64(value: string): boolean {
  try {
    base64ToBytes(value);
    return true;
  } catch {
    return false;
  }
}

function validateScript(
  value: unknown,
  issues: string[],
): AutomationScriptBundle | undefined {
  if (!isRecord(value)) {
    issues.push("spec.script: expected an object");
    return undefined;
  }

  const entrypoint = validateRequiredString(
    value,
    "entrypoint",
    "spec.script.entrypoint",
    issues,
  );
  const setupScriptPath = validateOptionalString(
    value,
    "setup_script_path",
    "spec.script.setup_script_path",
    issues,
  );
  const tarballBase64 = validateRequiredString(
    value,
    "tarball_base64",
    "spec.script.tarball_base64",
    issues,
  );
  if (tarballBase64 !== undefined && !decodesAsBase64(tarballBase64)) {
    issues.push("spec.script.tarball_base64: expected base64");
    return undefined;
  }

  if (!entrypoint || !tarballBase64) return undefined;

  return {
    entrypoint,
    ...(setupScriptPath !== undefined && {
      setup_script_path: setupScriptPath,
    }),
    tarball_base64: tarballBase64,
  };
}

export function serializeAutomation(a: Automation): AutomationExportFile {
  const timezone = a.timezone ?? a.trigger.timezone;
  // The file carries a single repository: the first one the automation clones.
  const [primaryRepository] = a.repositories ?? [];
  const repository = primaryRepository?.url ?? a.repository;
  const branch = primaryRepository ? primaryRepository.ref : a.branch;
  const spec: AutomationSpec = {
    name: a.name,
    trigger: { ...a.trigger },
    enabled: a.enabled,
    prompt: a.prompt,
    ...(repository !== undefined && { repository }),
    ...(a.model !== undefined && { model: a.model }),
    ...(a.timeout != null && { timeout: a.timeout }),
    ...(branch !== undefined && { branch }),
    ...(a.plugins !== undefined && { plugins: [...a.plugins] }),
    ...(a.notification !== undefined && { notification: a.notification }),
    ...(timezone !== undefined && { timezone }),
  };

  const envelope = getImportExportSpec();
  return { version: envelope.fileVersion, kind: envelope.fileKind, spec };
}

/**
 * The file of an automation that runs a script bundle instead of a prompt:
 * the same spec plus the bundle itself, so an import can upload it again.
 * `tarball` is what the service's tarball endpoint returned.
 */
export function serializeScriptAutomation(
  a: Automation,
  tarball: Uint8Array,
): AutomationExportFile {
  if (!a.entrypoint) {
    throw new Error("This automation has no entrypoint to export.");
  }

  const file = serializeAutomation(a);
  return {
    ...file,
    spec: {
      ...file.spec,
      script: {
        entrypoint: a.entrypoint,
        ...(a.setup_script_path
          ? { setup_script_path: a.setup_script_path }
          : {}),
        tarball_base64: bytesToBase64(tarball),
      },
    },
  };
}

export function getAutomationExportFilename(
  automation: Pick<Automation, "id" | "name">,
): string {
  const slug = automation.name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return `${slug || automation.id}${getImportExportSpec().filenameSuffix}`;
}

export function parseAutomationFile(json: unknown): AutomationSpec {
  const issues: string[] = [];
  if (!isRecord(json)) {
    throw new AutomationFileValidationError(["file: expected a JSON object"]);
  }

  const envelope = getImportExportSpec();
  if (json.version !== envelope.fileVersion) {
    issues.push(`version: expected ${envelope.fileVersion}`);
  }
  if (json.kind !== envelope.fileKind) {
    issues.push(`kind: expected "${envelope.fileKind}"`);
  }
  if (!isRecord(json.spec)) {
    issues.push("spec: expected an object");
    throw new AutomationFileValidationError(issues);
  }

  const name = validateRequiredString(json.spec, "name", "spec.name", issues);
  let prompt: string | undefined;
  let script: AutomationScriptBundle | undefined;
  if (json.spec.script == null) {
    prompt = validateRequiredString(json.spec, "prompt", "spec.prompt", issues);
  } else {
    // A script automation carries its bundle in place of a prompt.
    script = validateScript(json.spec.script, issues);
    if (json.spec.prompt != null) {
      issues.push("spec.prompt: must be omitted for a script automation");
    }
  }
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

  if (issues.length > 0 || !name || !trigger || (!prompt && !script)) {
    throw new AutomationFileValidationError(issues);
  }

  return {
    name,
    prompt: prompt ?? null,
    trigger,
    enabled: enabled as boolean,
    ...(script !== undefined && { script }),
    ...(repository !== undefined && { repository }),
    ...(model !== undefined && { model }),
    ...(timeout !== undefined && { timeout }),
    ...(branch !== undefined && { branch }),
    ...(plugins !== undefined && { plugins }),
    ...(notification !== undefined && { notification }),
    ...(timezone !== undefined && { timezone }),
  };
}
