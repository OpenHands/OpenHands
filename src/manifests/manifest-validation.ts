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

import {
  BUNDLE_CONFIG_FILENAME,
  SETUP_PLACEHOLDER_NAMESPACES,
  SETUP_VERSION,
} from "./types";

const ENTRY_ID_PATTERN = /^[a-z0-9]+(-[a-z0-9]+)*$/;
const FIELD_NAME_PATTERN = /^[a-z][A-Za-z0-9]*$/;
/** Copy must never be able to inject markup into the host. */
const MARKUP_PATTERN = /<[A-Za-z/!]/;
/**
 * A template version, declared as `version` by a prompt entry and as
 * `setup.bundle.version` by a bundle one. Sent to the service as template
 * provenance, so it is checked at admission. Full semver, as the spec at
 * semver.org states it: a catalog entry published with a pre-release or build
 * suffix is still a version this host may forward, and refusing one would drop
 * the entry from the registry outright.
 */
const TEMPLATE_VERSION_PATTERN =
  /^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:-(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*)?(?:\+[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*)?$/;
// The characters a command of plain words and paths is made of. No shell
// metacharacters, which the service rejects anyway and a bundle has no reason
// to need; where those words may point is `isPlainCommand`'s to say.
const BUNDLE_COMMAND_PATTERN = /^[A-Za-z0-9 ._/-]+$/;
const BUNDLE_PATH_PATTERN = /^[A-Za-z0-9._-]+(\/[A-Za-z0-9._-]+)*$/;
const BUNDLE_SOURCE_PATTERN =
  /^(skills|automations)\/[A-Za-z0-9._-]+(\/[A-Za-z0-9._-]+)*$/;
/** Every `{{` must open a known namespace and close immediately. */
const UNKNOWN_PLACEHOLDER_PATTERN = new RegExp(
  `\\{\\{(?!(?:${SETUP_PLACEHOLDER_NAMESPACES.join("|")})\\.[A-Za-z0-9_.]+\\}\\})`,
);

const SETUP_MODES = ["direct", "assisted"] as const;
const FIELD_TYPES = [
  "text",
  "textarea",
  "select",
  "number",
  "cron",
  "timezone",
  "repo-picker",
  "llm-profile",
  "event-source",
  "event-type",
  "plugin-sources",
  "tarball-upload",
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

/**
 * A relative path that stays inside the archive it names.
 *
 * The character class alone is not enough: `.` and `..` are made of allowed
 * characters, so a segment check is what actually keeps a packed file from
 * climbing out of the directory it is extracted into.
 */
function isRelativePath(value: unknown, pattern: RegExp): value is string {
  return (
    typeof value === "string" &&
    pattern.test(value) &&
    !value.split("/").some((segment) => segment === "." || segment === "..")
  );
}

/**
 * A command whose every word stays inside the archive it is run in.
 *
 * The character class is not what keeps it there: `/` and `.` are allowed
 * characters, so `/bin/sh setup.sh` and `python3 ../../etc/x.py` are made of
 * them and each names something outside the extracted directory. The segment
 * rule packed paths are held to is what refuses those, and requiring a word at
 * all is what refuses an entrypoint of nothing but spaces.
 */
function isPlainCommand(value: unknown): value is string {
  if (typeof value !== "string" || !BUNDLE_COMMAND_PATTERN.test(value)) {
    return false;
  }
  const words = value.split(" ").filter((word) => word.length > 0);
  return (
    words.length > 0 &&
    words.every(
      (word) =>
        !word.startsWith("/") &&
        !word.split("/").some((segment) => segment === ".."),
    )
  );
}

function checkRequires(check: SetupChecker, requires: unknown): void {
  if (!isRecord(requires)) {
    check.fail("requires", "must be an object");
    return;
  }

  const { integrations, features } = requires;
  // Empty is meaningful rather than malformed: an automation that needs no
  // credential connects to nothing, and the key stays required so that saying
  // so is a deliberate statement instead of an omission.
  if (!isRecord(integrations)) {
    check.fail("requires.integrations", "must be an object");
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

  const { minLength, maxLength, min, max, format } = constraints;
  if (minLength !== undefined && !isInteger(minLength, 0)) {
    check.fail(`${path}.minLength`, "must be a non-negative integer");
  }
  if (maxLength !== undefined && !isInteger(maxLength, 1)) {
    check.fail(`${path}.maxLength`, "must be a positive integer");
  }
  if (min !== undefined && !isInteger(min, 0)) {
    check.fail(`${path}.min`, "must be a non-negative integer");
  }
  if (max !== undefined && !isInteger(max, 1)) {
    check.fail(`${path}.max`, "must be a positive integer");
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
  if (
    field.default !== undefined &&
    typeof field.default !== "string" &&
    typeof field.default !== "number" &&
    typeof field.default !== "boolean" &&
    field.default !== null
  ) {
    check.fail(`${path}.default`, "must be a string, number, boolean or null");
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
  // The host branches on this key to decide whether a field's value is a list,
  // so a field declaring it anywhere else would be seeded with a list and
  // rendered as a string.
  if (
    field.multiple !== undefined &&
    (field.multiple !== true || type !== "repo-picker")
  ) {
    check.fail(
      `${path}.multiple`,
      "may only be true, and only on a repository field",
    );
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
function checkForm(
  check: SetupChecker,
  form: unknown,
): { kinds: string[]; fieldNames: string[] } {
  if (!isRecord(form)) {
    check.fail("setup.form", "must be an object");
    return { kinds: [], fieldNames: [] };
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

  return { kinds, fieldNames: names };
}

function checkActionFeatures(
  check: SetupChecker,
  features: unknown,
  path: string,
): void {
  if (!Array.isArray(features) || features.length === 0) {
    check.fail(path, "must be a non-empty array");
    return;
  }
  features.forEach((feature, index) => {
    if (typeof feature !== "string" || feature.length === 0) {
      check.fail(`${path}[${index}]`, "must be a non-empty string");
    }
  });
}

function checkAction(
  check: SetupChecker,
  kind: string,
  action: unknown,
  commonFieldNames: string[],
): void {
  const path = `setup.actions.${kind}`;
  if (!isRecord(action)) {
    check.fail(path, "must be an object");
    return;
  }

  check.copy(action.label, `${path}.label`);
  check.copy(action.help, `${path}.help`);
  checkActionFeatures(check, action.features, `${path}.features`);
  const actionFieldNames = checkFields(check, action.args, `${path}.args`);
  actionFieldNames
    .filter((name) => commonFieldNames.includes(name))
    .forEach((name) =>
      check.fail(`${path}.args.${name}`, "is already declared by the form"),
    );

  if (kind === "prompt") {
    check.templateValue(action.prompt, `${path}.prompt`);
  } else if (kind === "plugin") {
    check.templateValue(action.prompt, `${path}.prompt`);
    check.templateValue(action.plugins, `${path}.plugins`);
  } else if (kind === "upload") {
    check.templateValue(action.tarballPath, `${path}.tarballPath`);
    check.templateValue(action.entrypoint, `${path}.entrypoint`);
    if (action.setupScript !== undefined) {
      check.templateValue(action.setupScript, `${path}.setupScript`);
    }
  } else {
    check.fail(path, "is not a supported action kind");
  }

  const allowedByKind: Record<string, string[]> = {
    prompt: ["label", "help", "features", "args", "prompt"],
    plugin: ["label", "help", "features", "args", "prompt", "plugins"],
    upload: [
      "label",
      "help",
      "features",
      "args",
      "tarballPath",
      "entrypoint",
      "setupScript",
    ],
  };
  const allowed = allowedByKind[kind] ?? [];
  Object.keys(action)
    .filter((key) => !allowed.includes(key))
    .forEach((key) => check.fail(`${path}.${key}`, "is not allowed"));
}

function checkActions(
  check: SetupChecker,
  actions: unknown,
  commonFieldNames: string[],
): void {
  if (!isRecord(actions) || Object.keys(actions).length === 0) {
    check.fail("setup.actions", "must be a non-empty object");
    return;
  }
  Object.entries(actions).forEach(([kind, action]) =>
    checkAction(check, kind, action, commonFieldNames),
  );
}

function fieldsContainType(fields: unknown, type: string): boolean {
  if (!isRecord(fields)) return false;
  return Object.values(fields).some(
    (field) => isRecord(field) && field.type === type,
  );
}

function formContainsFieldType(form: unknown, type: string): boolean {
  if (!isRecord(form)) return false;
  const triggerGroups = isRecord(form.triggers)
    ? Object.values(form.triggers)
    : [];
  return (
    fieldsContainType(form.args, type) ||
    triggerGroups.some((fields) => fieldsContainType(fields, type))
  );
}

function actionsContainRepoPicker(actions: unknown): boolean {
  if (!isRecord(actions)) return false;
  return Object.values(actions).some(
    (action) =>
      isRecord(action) && fieldsContainType(action.args, "repo-picker"),
  );
}

function setupSuppliesEventSource(setup: Rec): boolean {
  const eventFields = isRecord(setup.form)
    ? isRecord(setup.form.triggers)
      ? setup.form.triggers.event
      : undefined
    : undefined;
  return (
    fieldsContainType(eventFields, "event-source") ||
    formContainsFieldType(setup.form, "repo-picker") ||
    actionsContainRepoPicker(setup.actions)
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

/**
 * The script tarball a direct entry may ship.
 *
 * The strings here are commands and paths this host acts on, so they are held
 * to a closed character set rather than the placeholder rules copy uses: an
 * entrypoint with a shell metacharacter, or a packed path that escapes the
 * archive, is refused here rather than by the service.
 */
function checkBundle(check: SetupChecker, bundle: unknown): void {
  if (!isRecord(bundle)) {
    check.fail("setup.bundle", "must be an object");
    return;
  }

  if (
    typeof bundle.version !== "string" ||
    !TEMPLATE_VERSION_PATTERN.test(bundle.version)
  ) {
    check.fail("setup.bundle.version", "must be a semantic version");
  }
  if (!isPlainCommand(bundle.entrypoint)) {
    check.fail(
      "setup.bundle.entrypoint",
      "must be a plain command that stays inside the archive",
    );
  }
  if (
    bundle.setupScript !== undefined &&
    !isRelativePath(bundle.setupScript, BUNDLE_PATH_PATTERN)
  ) {
    check.fail("setup.bundle.setupScript", "must be a relative path");
  }
  if (bundle.timeout !== undefined && !isInteger(bundle.timeout, 1)) {
    check.fail("setup.bundle.timeout", "must be a positive integer");
  }

  if (!isRecord(bundle.files) || Object.keys(bundle.files).length === 0) {
    check.fail("setup.bundle.files", "must be a non-empty object");
  } else {
    Object.entries(bundle.files).forEach(([packedPath, source]) => {
      if (!isRelativePath(packedPath, BUNDLE_PATH_PATTERN)) {
        check.fail(`setup.bundle.files.${packedPath}`, "is not a packed path");
      }
      // The rendered config is packed under this name too, and a tar carrying
      // the name twice leaves which one the script reads to whichever
      // extractor unpacks it.
      if (packedPath === BUNDLE_CONFIG_FILENAME) {
        check.fail(
          `setup.bundle.files.${packedPath}`,
          "is the name the rendered config is packed under",
        );
      }
      if (!isRelativePath(source, BUNDLE_SOURCE_PATTERN)) {
        check.fail(
          `setup.bundle.files.${packedPath}`,
          "must name a file under skills/ or automations/",
        );
      }
    });

    // A setup script the archive does not carry is a create request the
    // service accepts and the first run fails on, and it is also the only
    // thing packed executable - naming an unpacked file makes that rule
    // unreachable.
    if (
      typeof bundle.setupScript === "string" &&
      !(bundle.setupScript in bundle.files)
    ) {
      check.fail("setup.bundle.setupScript", "must name a packed file");
    }
  }

  if (!isRecord(bundle.config) || Object.keys(bundle.config).length === 0) {
    check.fail("setup.bundle.config", "must be a non-empty object");
  } else {
    checkBundleConfig(check, bundle.config, "setup.bundle.config");
  }
}

/** Every string leaf of the config is a payload value: placeholders, no markup rule. */
function checkBundleConfig(
  check: SetupChecker,
  node: unknown,
  path: string,
): void {
  if (typeof node === "string") {
    check.templateValue(node, path);
    return;
  }
  if (Array.isArray(node)) {
    node.forEach((item, index) =>
      checkBundleConfig(check, item, `${path}[${index}]`),
    );
    return;
  }
  if (isRecord(node)) {
    Object.entries(node).forEach(([key, value]) =>
      checkBundleConfig(check, value, `${path}.${key}`),
    );
    return;
  }
  if (node !== null && typeof node !== "number" && typeof node !== "boolean") {
    check.fail(
      path,
      "must be a string, number, boolean, null, array or object",
    );
  }
}

function checkMode(
  check: SetupChecker,
  setup: Rec,
  kinds: string[],
  commonFieldNames: string[],
): void {
  if (!isOneOf(setup.mode, SETUP_MODES)) {
    check.fail("setup.mode", "is not a supported mode");
    return;
  }

  if (setup.mode === "direct") {
    // A direct entry produces one of three things: a prompt, a script bundle
    // the host packs and uploads, or a selectable set of action variants.
    // More than one would be ambiguous, and none is nothing to create.
    const hasPrompt = setup.prompt !== undefined;
    const hasBundle = setup.bundle !== undefined;
    const hasActions = setup.actions !== undefined;
    const variants = [hasPrompt, hasBundle, hasActions].filter(Boolean);
    if (variants.length !== 1) {
      check.fail(
        "setup",
        "must declare exactly one of prompt, bundle or actions for direct setup",
      );
    } else if (hasBundle) {
      checkBundle(check, setup.bundle);
    } else if (hasActions) {
      checkActions(check, setup.actions, commonFieldNames);
    } else {
      check.templateValue(setup.prompt, "setup.prompt");
    }
    // A direct entry may carry a fallback-conversation seed for deployments
    // that cannot run the direct path, held to the same rules as an assisted
    // message.
    if (setup.message !== undefined) checkMessage(check, setup.message);

    if (kinds.length === 0) {
      check.fail(
        "setup.form.triggers",
        "must declare at least one trigger kind",
      );
    }
    if (kinds.includes("event") && !setupSuppliesEventSource(setup)) {
      check.fail(
        "setup.form.triggers.event",
        "must declare an event source field or repository picker",
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
  check.absent(setup, "bundle", "setup", "is only allowed for direct setup");
  check.absent(setup, "actions", "setup", "is only allowed for direct setup");
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
    candidate.version !== undefined &&
    (typeof candidate.version !== "string" ||
      !TEMPLATE_VERSION_PATTERN.test(candidate.version))
  ) {
    check.fail("version", "must be a semantic version");
  }
  if (
    candidate.skill !== undefined &&
    (typeof candidate.skill !== "string" ||
      !ENTRY_ID_PATTERN.test(candidate.skill))
  ) {
    check.fail("skill", "must be a lowercase slug");
  }

  checkRequires(check, candidate.requires);
  const { kinds, fieldNames } = checkForm(check, setup.form);
  checkMode(check, setup, kinds, fieldNames);

  return { valid: check.errors.length === 0, errors: check.errors };
}
