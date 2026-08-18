/**
 * Admission policy for extension-authored catalog manifests.
 *
 * A manifest is data authored in a different repository that instructs this host
 * to render copy and compose a request. The host therefore decides what a
 * manifest is *permitted* to do rather than trusting the file — validation here
 * is a trust boundary, not a convenience check, and it deliberately does not
 * defer to a schema shipped alongside the manifests it would be validating.
 *
 * Beyond mirroring the published schema, this adds the invariants the host's own
 * derivation depends on: a single trigger kind to read, a repository field for
 * an event trigger's source, and field names unique across the form. Without
 * them the derived request body would be silently wrong rather than rejected.
 *
 * A manifest that fails any check is rejected outright. It never renders a
 * partial UI, because everything downstream treats its content as instructions.
 */

import { SETUP_PLACEHOLDER_NAMESPACES, SETUP_VERSION } from "./types";

const ENTRY_ID_PATTERN = /^[a-z0-9]+(-[a-z0-9]+)*$/;
const FIELD_NAME_PATTERN = /^[a-z][A-Za-z0-9]*$/;
/** Copy must never be able to inject markup into the host. */
const MARKUP_PATTERN = /<[A-Za-z/!]/;
/** Every `{{` must open a known namespace and close immediately. */
const UNKNOWN_PLACEHOLDER_PATTERN = new RegExp(
  `\\{\\{(?!(?:${SETUP_PLACEHOLDER_NAMESPACES.join("|")})\\.[A-Za-z0-9_.]+\\}\\})`,
);

const SETUP_MODES = ["direct", "assisted"] as const;
const FIELD_TYPES = [
  "text",
  "textarea",
  "select",
  "cron",
  "timezone",
  "repo-picker",
] as const;
const GIT_PROVIDERS = ["github", "gitlab", "bitbucket"] as const;
const TRIGGER_KINDS = ["cron", "event"] as const;
const CONSTRAINT_FORMATS = ["safeExpressionLiteral"] as const;

/** Setup context only, so this can never become a channel for runtime instructions. */
const MAX_MESSAGE_LENGTH = 2000;

export interface SetupValidationResult {
  valid: boolean;
  errors: string[];
}

type Rec = Record<string, unknown>;

function isRecord(value: unknown): value is Rec {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isOneOf<T extends readonly string[]>(
  value: unknown,
  allowed: T,
): value is T[number] {
  return (
    typeof value === "string" && (allowed as readonly string[]).includes(value)
  );
}

function isInteger(value: unknown, minimum: number): boolean {
  return (
    typeof value === "number" && Number.isInteger(value) && value >= minimum
  );
}

class SetupChecker {
  readonly errors: string[] = [];

  fail(path: string, reason: string): false {
    this.errors.push(`${path}: ${reason}`);
    return false;
  }

  /** Literal user-visible copy. Carries no markup. */
  copy(value: unknown, path: string): boolean {
    if (typeof value !== "string" || value.length === 0) {
      return this.fail(path, "must be a non-empty string");
    }
    if (MARKUP_PATTERN.test(value)) {
      return this.fail(path, "must not contain markup");
    }
    return true;
  }

  placeholders(value: string, path: string): boolean {
    if (UNKNOWN_PLACEHOLDER_PATTERN.test(value)) {
      return this.fail(path, "uses an unknown placeholder namespace");
    }
    return true;
  }

  /** Copy that may embed `{{namespace.key}}` placeholders. */
  templateCopy(value: unknown, path: string): boolean {
    return this.copy(value, path) && this.placeholders(value as string, path);
  }

  /**
   * A request-body string that may embed placeholders. Never rendered, so
   * expression syntax such as a JMESPath filter is allowed through.
   */
  templateValue(value: unknown, path: string): boolean {
    if (typeof value !== "string" || value.length === 0) {
      return this.fail(path, "must be a non-empty string");
    }
    return this.placeholders(value, path);
  }

  absent(container: Rec, key: string, path: string, reason: string): boolean {
    if (key in container) return this.fail(`${path}.${key}`, reason);
    return true;
  }
}

function checkRequires(check: SetupChecker, requires: unknown): void {
  if (!isRecord(requires)) {
    check.fail("requires", "must be an object");
    return;
  }

  const { integrations, features } = requires;
  if (!isRecord(integrations) || Object.keys(integrations).length === 0) {
    check.fail("requires.integrations", "must be a non-empty object");
  } else {
    Object.entries(integrations).forEach(([id, requirement]) => {
      const path = `requires.integrations.${id}`;
      if (!isRecord(requirement)) {
        check.fail(path, "must be an object");
        return;
      }
      check.copy(requirement.message, `${path}.message`);
      if ("required" in requirement && requirement.required !== false) {
        check.fail(`${path}.required`, "may only be false");
      }
      const allowed = ["message", "required"];
      Object.keys(requirement)
        .filter((key) => !allowed.includes(key))
        .forEach((key) => check.fail(`${path}.${key}`, "is not allowed"));
    });
  }

  if (features !== undefined) {
    if (!Array.isArray(features) || features.length === 0) {
      check.fail("requires.features", "must be a non-empty array");
    } else {
      features.forEach((feature, index) => {
        if (typeof feature !== "string" || feature.length === 0) {
          check.fail(
            `requires.features[${index}]`,
            "must be a non-empty string",
          );
        }
      });
    }
  }
}

function checkConstraints(
  check: SetupChecker,
  constraints: unknown,
  path: string,
): void {
  if (!isRecord(constraints) || Object.keys(constraints).length === 0) {
    check.fail(path, "must be a non-empty object");
    return;
  }

  const { minLength, maxLength, format } = constraints;
  if (minLength !== undefined && !isInteger(minLength, 0)) {
    check.fail(`${path}.minLength`, "must be a non-negative integer");
  }
  if (maxLength !== undefined && !isInteger(maxLength, 1)) {
    check.fail(`${path}.maxLength`, "must be a positive integer");
  }
  if (format !== undefined && !isOneOf(format, CONSTRAINT_FORMATS)) {
    check.fail(`${path}.format`, "is not a supported format");
  }
}

function checkField(check: SetupChecker, field: unknown, path: string): void {
  if (!isRecord(field)) {
    check.fail(path, "must be an object");
    return;
  }

  const { type, label, help, placeholder, required, provider, options } = field;

  if (!isOneOf(type, FIELD_TYPES)) {
    check.fail(`${path}.type`, "is not a supported field type");
  }
  check.copy(label, `${path}.label`);
  check.copy(help, `${path}.help`);
  if (placeholder !== undefined) check.copy(placeholder, `${path}.placeholder`);
  if (field.default !== undefined && typeof field.default !== "string") {
    check.fail(`${path}.default`, "must be a string");
  }
  if (typeof required !== "boolean") {
    check.fail(`${path}.required`, "must be a boolean");
  }
  if (provider !== undefined && !isOneOf(provider, GIT_PROVIDERS)) {
    check.fail(`${path}.provider`, "is not a supported provider");
  }
  if (type === "repo-picker" && provider === undefined) {
    check.fail(`${path}.provider`, "is required for a repository field");
  }

  if (options !== undefined) {
    if (type !== "select") {
      check.fail(`${path}.options`, "is only allowed on a select field");
    } else if (!Array.isArray(options) || options.length === 0) {
      check.fail(`${path}.options`, "must be a non-empty array");
    } else {
      options.forEach((option, index) => {
        const optionPath = `${path}.options[${index}]`;
        if (!isRecord(option)) {
          check.fail(optionPath, "must be an object");
          return;
        }
        if (typeof option.value !== "string" || option.value.length === 0) {
          check.fail(`${optionPath}.value`, "must be a non-empty string");
        }
        check.copy(option.label, `${optionPath}.label`);
      });
    }
  }

  if (field.constraints !== undefined) {
    checkConstraints(check, field.constraints, `${path}.constraints`);
  }
}

/** Returns the declared field names, so the caller can check them for collisions. */
function checkFields(
  check: SetupChecker,
  fields: unknown,
  path: string,
): string[] {
  if (!isRecord(fields) || Object.keys(fields).length === 0) {
    check.fail(path, "must be a non-empty object");
    return [];
  }

  return Object.entries(fields).map(([name, field]) => {
    if (!FIELD_NAME_PATTERN.test(name)) {
      check.fail(`${path}.${name}`, "is not a valid field name");
    }
    checkField(check, field, `${path}.${name}`);
    return name;
  });
}

/** Returns the trigger kinds declared, in order. */
function checkForm(check: SetupChecker, form: unknown): string[] {
  if (!isRecord(form)) {
    check.fail("setup.form", "must be an object");
    return [];
  }

  if (form.note !== undefined) check.copy(form.note, "setup.form.note");

  const names = checkFields(check, form.args, "setup.form.args");
  const kinds: string[] = [];

  if (form.triggers !== undefined) {
    if (!isRecord(form.triggers) || Object.keys(form.triggers).length === 0) {
      check.fail("setup.form.triggers", "must be a non-empty object");
    } else {
      Object.entries(form.triggers).forEach(([kind, fields]) => {
        if (!isOneOf(kind, TRIGGER_KINDS)) {
          check.fail(`setup.form.triggers.${kind}`, "is not a trigger kind");
          return;
        }
        kinds.push(kind);
        names.push(
          ...checkFields(check, fields, `setup.form.triggers.${kind}`),
        );
      });
    }
  }

  // The host merges both halves into one value map, so a repeated name would
  // silently shadow a field and misaddress every error reported against it.
  const duplicates = names.filter(
    (name, index) => names.indexOf(name) !== index,
  );
  Array.from(new Set(duplicates)).forEach((name) =>
    check.fail(`setup.form.${name}`, "is declared more than once"),
  );

  return kinds;
}

function hasRepoPicker(form: unknown): boolean {
  if (!isRecord(form)) return false;
  const groups = [
    form.args,
    ...Object.values(isRecord(form.triggers) ? form.triggers : {}),
  ];
  return groups.some(
    (group) =>
      isRecord(group) &&
      Object.values(group).some(
        (field) => isRecord(field) && field.type === "repo-picker",
      ),
  );
}

function checkMessage(check: SetupChecker, message: unknown): void {
  if (check.templateCopy(message, "setup.message")) {
    if ((message as string).length > MAX_MESSAGE_LENGTH) {
      check.fail(
        "setup.message",
        `must be at most ${MAX_MESSAGE_LENGTH} characters`,
      );
    }
  }
}

function checkMode(check: SetupChecker, setup: Rec, kinds: string[]): void {
  if (!isOneOf(setup.mode, SETUP_MODES)) {
    check.fail("setup.mode", "is not a supported mode");
    return;
  }

  if (setup.mode === "direct") {
    check.templateValue(setup.prompt, "setup.prompt");
    // A direct entry may carry a fallback-conversation seed for deployments
    // that cannot run the direct path, held to the same rules as an assisted
    // message.
    if (setup.message !== undefined) checkMessage(check, setup.message);

    // The derivation reads a single trigger kind, and an event trigger takes
    // its source from the repository field's provider.
    if (kinds.length !== 1) {
      check.fail(
        "setup.form.triggers",
        "must declare exactly one trigger kind",
      );
    }
    if (kinds.includes("event") && !hasRepoPicker(setup.form)) {
      check.fail(
        "setup.form",
        "must declare a repository field for an event trigger",
      );
    }
    if (setup.filter !== undefined) {
      if (!kinds.includes("event")) {
        check.fail("setup.filter", "is only allowed for an event trigger");
      } else {
        check.templateValue(setup.filter, "setup.filter");
      }
    }
    return;
  }

  checkMessage(check, setup.message);
  check.absent(setup, "prompt", "setup", "is only allowed for direct setup");
  check.absent(setup, "filter", "setup", "is only allowed for direct setup");
}

/** Whether a catalog entry ships a setup experience at all. It is optional. */
export function hasSetupBlock(candidate: unknown): boolean {
  return isRecord(candidate) && candidate.setup !== undefined;
}

/**
 * Decide whether this host will act on a catalog manifest.
 *
 * The version is checked first and fails closed: a format this host does not
 * recognise is refused rather than interpreted with today's rules.
 */
export function validateSetupEntry(candidate: unknown): SetupValidationResult {
  const check = new SetupChecker();

  if (!isRecord(candidate)) {
    return { valid: false, errors: ["manifest: must be an object"] };
  }

  const { setup } = candidate;
  if (!isRecord(setup)) {
    return { valid: false, errors: ["setup: must be an object"] };
  }
  if (setup.version !== SETUP_VERSION) {
    return {
      valid: false,
      errors: [`setup.version: must be "${SETUP_VERSION}"`],
    };
  }

  if (
    typeof candidate.id !== "string" ||
    !ENTRY_ID_PATTERN.test(candidate.id)
  ) {
    check.fail("id", "must be a lowercase slug");
  }
  check.copy(candidate.name, "name");
  check.copy(candidate.description, "description");
  if (
    candidate.skill !== undefined &&
    (typeof candidate.skill !== "string" ||
      !ENTRY_ID_PATTERN.test(candidate.skill))
  ) {
    check.fail("skill", "must be a lowercase slug");
  }

  checkRequires(check, candidate.requires);
  const kinds = checkForm(check, setup.form);
  checkMode(check, setup, kinds);

  return { valid: check.errors.length === 0, errors: check.errors };
}
