/**
 * Admission policy for extension manifests.
 *
 * A manifest is data authored in a different repository that instructs this host
 * to make requests and render copy. The host therefore decides what a manifest
 * is *permitted* to do rather than trusting the file — validation here is a
 * trust boundary, not a convenience check, and it deliberately does not defer to
 * a schema shipped alongside the manifests it would be validating.
 *
 * A manifest that fails any check is rejected outright. It never renders a
 * partial UI, because everything downstream treats manifest content as
 * instructions.
 */

import {
  MANIFEST_ACTIONS,
  MANIFEST_PLACEHOLDER_NAMESPACES,
  MANIFEST_VERSION,
  type ExtensionManifest,
  type ManifestAction,
} from "./types";

/** Service-relative path. The deployment base path is resolved by the host. */
const SERVICE_PATH_PATTERN = /^\/v1\/[A-Za-z0-9/_-]*$/;
const ROUTE_PATH_PATTERN = /^\/[A-Za-z0-9/_-]*$/;
const FIELD_NAME_PATTERN = /^[a-z][A-Za-z0-9]*$/;
const SECRET_KEY_PATTERN = /^[A-Z][A-Z0-9_]*$/;
const ANALYTICS_ID_PATTERN = /^[a-z][a-z0-9_]*$/;
const ANALYTICS_PROPERTY_PATTERN = /^[a-z][a-z0-9_]*$/;
/** Copy must never be able to inject markup into the host. */
const MARKUP_PATTERN = /<[A-Za-z/!]/;
/** Every `{{` must open a known namespace and close immediately. */
const UNKNOWN_PLACEHOLDER_PATTERN = new RegExp(
  `\\{\\{(?!(?:${MANIFEST_PLACEHOLDER_NAMESPACES.join("|")})\\.[A-Za-z0-9_.]+\\}\\})`,
);

const FIELD_TYPES = [
  "text",
  "textarea",
  "select",
  "cron",
  "repo-picker",
] as const;
const GIT_PROVIDERS = ["github", "gitlab", "bitbucket"] as const;
const TRIGGER_KINDS = ["cron", "event"] as const;
const CONSTRAINT_FORMATS = ["safeExpressionLiteral"] as const;
const BINDING_CONSTRAINTS = ["options", "minIntervalSeconds"] as const;
const PREFLIGHT_RUN_ON = ["fieldBlur", "beforeSubmit"] as const;
const ERROR_TARGETS = ["field", "form"] as const;
const ANALYTICS_EVENTS = [
  "route.entered",
  "capabilities.resolved",
  "validation.succeeded",
  "submit.succeeded",
  "submit.failed",
] as const;

/** Setup context only, so this can never become a channel for runtime instructions. */
const MAX_ASSISTED_MESSAGE_LENGTH = 2000;

export interface ManifestValidationResult {
  valid: boolean;
  errors: string[];
}

type Rec = Record<string, unknown>;

function isRecord(value: unknown): value is Rec {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNonEmptyArray(value: unknown): value is unknown[] {
  return Array.isArray(value) && value.length > 0;
}

function isOneOf<T extends readonly string[]>(
  value: unknown,
  allowed: T,
): value is T[number] {
  return (
    typeof value === "string" && (allowed as readonly string[]).includes(value)
  );
}

class ManifestChecker {
  readonly errors: string[] = [];

  fail(path: string, reason: string): false {
    this.errors.push(`${path}: ${reason}`);
    return false;
  }

  /** Literal user-visible copy. Carries no markup and no placeholders. */
  copy(value: unknown, path: string): boolean {
    if (typeof value !== "string" || value.length === 0) {
      return this.fail(path, "must be a non-empty string");
    }
    if (MARKUP_PATTERN.test(value)) {
      return this.fail(path, "must not contain markup");
    }
    return true;
  }

  /** Copy that may embed `{{namespace.key}}` placeholders. */
  templateCopy(value: unknown, path: string): boolean {
    return this.copy(value, path) && this.placeholders(value as string, path);
  }

  placeholders(value: string, path: string): boolean {
    if (UNKNOWN_PLACEHOLDER_PATTERN.test(value)) {
      return this.fail(path, "uses an unknown placeholder namespace");
    }
    return true;
  }

  servicePath(value: unknown, path: string): boolean {
    if (typeof value !== "string" || !SERVICE_PATH_PATTERN.test(value)) {
      return this.fail(path, "must be a service-relative /v1 path");
    }
    return true;
  }

  /** In-app destination only, so a manifest cannot navigate the user off-site. */
  inAppPath(value: unknown, path: string): boolean {
    if (!this.templateCopy(value, path)) return false;
    const destination = value as string;
    if (!destination.startsWith("/") || destination.startsWith("//")) {
      return this.fail(path, "must be an in-app path");
    }
    if (destination.includes("://")) {
      return this.fail(path, "must not contain a scheme");
    }
    return true;
  }

  /** Request-body values may nest, but there is nowhere to put code. */
  payloadValue(value: unknown, path: string): boolean {
    if (typeof value === "string") {
      return value.length > 0
        ? this.placeholders(value, path)
        : this.fail(path, "must be a non-empty string");
    }
    if (
      typeof value === "number" ||
      typeof value === "boolean" ||
      value === null
    ) {
      return true;
    }
    if (Array.isArray(value)) {
      return value.every((item, index) =>
        this.payloadValue(item, `${path}[${index}]`),
      );
    }
    if (isRecord(value)) {
      return Object.entries(value).every(([key, item]) =>
        this.payloadValue(item, `${path}.${key}`),
      );
    }
    return this.fail(path, "is not a valid request-body value");
  }

  requestBody(value: unknown, path: string): boolean {
    if (!isRecord(value) || Object.keys(value).length === 0) {
      return this.fail(path, "must be a non-empty object");
    }
    return this.payloadValue(value, path);
  }
}

function checkRoutes(check: ManifestChecker, routes: unknown): void {
  if (!isNonEmptyArray(routes)) {
    check.fail("routes", "must be a non-empty array");
    return;
  }
  routes.forEach((route, index) => {
    const path = `routes[${index}]`;
    if (!isRecord(route)) {
      check.fail(path, "must be an object");
      return;
    }
    if (
      typeof route.path !== "string" ||
      !ROUTE_PATH_PATTERN.test(route.path)
    ) {
      check.fail(`${path}.path`, "must be an in-app path");
    }
    if (route.page !== "setup") {
      check.fail(`${path}.page`, "names a page type this host cannot render");
    }
  });
}

function checkField(
  check: ManifestChecker,
  field: unknown,
  path: string,
): void {
  if (!isRecord(field)) {
    check.fail(path, "must be an object");
    return;
  }
  if (typeof field.name !== "string" || !FIELD_NAME_PATTERN.test(field.name)) {
    check.fail(`${path}.name`, "must be a camelCase identifier");
  }
  if (!isOneOf(field.type, FIELD_TYPES)) {
    check.fail(`${path}.type`, "names a field type this host cannot render");
  }
  check.copy(field.label, `${path}.label`);
  check.copy(field.help, `${path}.help`);
  if (typeof field.required !== "boolean") {
    check.fail(`${path}.required`, "must be a boolean");
  }
  if (field.placeholder !== undefined) {
    check.copy(field.placeholder, `${path}.placeholder`);
  }
  if (field.default !== undefined && typeof field.default !== "string") {
    check.fail(`${path}.default`, "must be a string");
  }

  if (field.type === "repo-picker" && !isOneOf(field.provider, GIT_PROVIDERS)) {
    check.fail(
      `${path}.provider`,
      "is required and must name a known provider",
    );
  }
  if (field.type !== "select" && field.options !== undefined) {
    check.fail(`${path}.options`, "is only allowed on select fields");
  }
  if (field.options !== undefined) {
    if (!isNonEmptyArray(field.options)) {
      check.fail(`${path}.options`, "must be a non-empty array");
    } else {
      field.options.forEach((option, index) => {
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
    const constraints = field.constraints;
    const constraintsPath = `${path}.constraints`;
    if (!isRecord(constraints)) {
      check.fail(constraintsPath, "must be an object");
      return;
    }
    if (
      constraints.minLength !== undefined &&
      typeof constraints.minLength !== "number"
    ) {
      check.fail(`${constraintsPath}.minLength`, "must be a number");
    }
    if (
      constraints.maxLength !== undefined &&
      typeof constraints.maxLength !== "number"
    ) {
      check.fail(`${constraintsPath}.maxLength`, "must be a number");
    }
    if (
      constraints.format !== undefined &&
      !isOneOf(constraints.format, CONSTRAINT_FORMATS)
    ) {
      check.fail(
        `${constraintsPath}.format`,
        "names a format check this host does not implement",
      );
    }
  }
}

function checkForm(check: ManifestChecker, form: unknown): void {
  if (!isRecord(form)) {
    check.fail("form", "must be an object");
    return;
  }
  if (form.note !== undefined) check.copy(form.note, "form.note");
  if (!isNonEmptyArray(form.fields)) {
    check.fail("form.fields", "must be a non-empty array");
    return;
  }
  form.fields.forEach((field, index) =>
    checkField(check, field, `form.fields[${index}]`),
  );
}

function checkCapabilities(
  check: ManifestChecker,
  capabilities: unknown,
): void {
  if (!isRecord(capabilities)) {
    check.fail("capabilities", "must be an object");
    return;
  }

  const discovery = capabilities.discovery;
  if (!isRecord(discovery)) {
    check.fail("capabilities.discovery", "must be an object");
  } else {
    if (discovery.method !== "GET") {
      check.fail("capabilities.discovery.method", "must be GET");
    }
    check.servicePath(discovery.path, "capabilities.discovery.path");
  }

  const requires = capabilities.requires;
  if (!isRecord(requires) || Object.keys(requires).length === 0) {
    check.fail("capabilities.requires", "must be a non-empty object");
  } else {
    if (
      requires.triggerKinds !== undefined &&
      !(
        Array.isArray(requires.triggerKinds) &&
        requires.triggerKinds.every((kind) => isOneOf(kind, TRIGGER_KINDS))
      )
    ) {
      check.fail(
        "capabilities.requires.triggerKinds",
        "names an unknown trigger kind",
      );
    }
    if (requires.ready !== undefined && requires.ready !== true) {
      check.fail("capabilities.requires.ready", "must be true when present");
    }
  }

  if (capabilities.bindings !== undefined) {
    if (!Array.isArray(capabilities.bindings)) {
      check.fail("capabilities.bindings", "must be an array");
    } else {
      capabilities.bindings.forEach((binding, index) => {
        const path = `capabilities.bindings[${index}]`;
        if (!isRecord(binding)) {
          check.fail(path, "must be an object");
          return;
        }
        if (typeof binding.field !== "string" || binding.field.length === 0) {
          check.fail(`${path}.field`, "must name a form field");
        }
        if (!isOneOf(binding.constraint, BINDING_CONSTRAINTS)) {
          check.fail(
            `${path}.constraint`,
            "names a constraint this host does not implement",
          );
        }
        if (typeof binding.from !== "string" || binding.from.length === 0) {
          check.fail(`${path}.from`, "must be a dotted capabilities path");
        }
      });
    }
  }

  const onUnsupported = capabilities.onUnsupported;
  if (!isRecord(onUnsupported)) {
    check.fail("capabilities.onUnsupported", "must be an object");
    return;
  }
  if (onUnsupported.behavior !== "block") {
    check.fail("capabilities.onUnsupported.behavior", "must be block");
  }
  check.copy(onUnsupported.message, "capabilities.onUnsupported.message");
}

function checkRequires(check: ManifestChecker, requires: unknown): void {
  if (!isRecord(requires)) {
    check.fail("requires", "must be an object");
    return;
  }

  if (!Array.isArray(requires.integrations)) {
    check.fail("requires.integrations", "must be an array");
  } else {
    requires.integrations.forEach((integration, index) => {
      const path = `requires.integrations[${index}]`;
      if (!isRecord(integration)) {
        check.fail(path, "must be an object");
        return;
      }
      if (typeof integration.id !== "string" || integration.id.length === 0) {
        check.fail(`${path}.id`, "must be a non-empty string");
      }
      check.copy(integration.reason, `${path}.reason`);
      if (!isOneOf(integration.enforcement, ["block", "warn"] as const)) {
        check.fail(`${path}.enforcement`, "must be block or warn");
      }
    });
  }

  if (!Array.isArray(requires.secrets)) {
    check.fail("requires.secrets", "must be an array");
  } else {
    requires.secrets.forEach((secret, index) => {
      const path = `requires.secrets[${index}]`;
      if (!isRecord(secret)) {
        check.fail(path, "must be an object");
        return;
      }
      if (
        typeof secret.key !== "string" ||
        !SECRET_KEY_PATTERN.test(secret.key)
      ) {
        check.fail(`${path}.key`, "must be an upper-case credential name");
      }
      check.copy(secret.label, `${path}.label`);
      check.copy(secret.help, `${path}.help`);
      if (typeof secret.required !== "boolean") {
        check.fail(`${path}.required`, "must be a boolean");
      }
      // The credential name is all a manifest may carry. Anything that looks
      // like a value is a manifest authoring mistake with a security cost.
      const extraKeys = Object.keys(secret).filter(
        (key) => !["key", "label", "help", "required"].includes(key),
      );
      if (extraKeys.length > 0) {
        check.fail(path, `must not carry ${extraKeys.join(", ")}`);
      }
    });
  }

  const onUnmet = requires.onUnmet;
  if (!isRecord(onUnmet)) {
    check.fail("requires.onUnmet", "must be an object");
  } else {
    if (onUnmet.behavior !== "block") {
      check.fail("requires.onUnmet.behavior", "must be block");
    }
    check.copy(onUnmet.message, "requires.onUnmet.message");
  }

  if (requires.onWarn !== undefined) {
    const onWarn = requires.onWarn;
    if (!isRecord(onWarn)) {
      check.fail("requires.onWarn", "must be an object");
    } else {
      if (onWarn.behavior !== "continue") {
        check.fail("requires.onWarn.behavior", "must be continue");
      }
      check.copy(onWarn.message, "requires.onWarn.message");
    }
  }
}

function checkValidation(check: ManifestChecker, validation: unknown): void {
  if (!isRecord(validation)) {
    check.fail("validation", "must be an object");
    return;
  }

  if (validation.preflight !== undefined) {
    const preflight = validation.preflight;
    if (!isRecord(preflight)) {
      check.fail("validation.preflight", "must be an object");
    } else {
      if (preflight.method !== "POST") {
        check.fail("validation.preflight.method", "must be POST");
      }
      check.servicePath(preflight.path, "validation.preflight.path");
      if (
        !isNonEmptyArray(preflight.runOn) ||
        !preflight.runOn.every((trigger) => isOneOf(trigger, PREFLIGHT_RUN_ON))
      ) {
        check.fail("validation.preflight.runOn", "names an unknown trigger");
      }
      if (
        preflight.debounceMs !== undefined &&
        typeof preflight.debounceMs !== "number"
      ) {
        check.fail("validation.preflight.debounceMs", "must be a number");
      }
      check.requestBody(preflight.body, "validation.preflight.body");
    }
  }

  const onInvalid = validation.onInvalid;
  if (!isRecord(onInvalid)) {
    check.fail("validation.onInvalid", "must be an object");
    return;
  }
  if (onInvalid.behavior !== "blockSubmit") {
    check.fail("validation.onInvalid.behavior", "must be blockSubmit");
  }
  if (!isOneOf(onInvalid.errorTarget, ERROR_TARGETS)) {
    check.fail("validation.onInvalid.errorTarget", "must be field or form");
  }
  if (onInvalid.errorMap !== undefined) {
    checkErrorMap(check, onInvalid.errorMap, "validation.onInvalid.errorMap");
  }
}

function checkErrorMap(
  check: ManifestChecker,
  errorMap: unknown,
  path: string,
): void {
  if (!isRecord(errorMap) || Object.keys(errorMap).length === 0) {
    check.fail(path, "must be a non-empty object");
    return;
  }
  Object.entries(errorMap).forEach(([key, target]) => {
    const isFieldName = (name: unknown) =>
      typeof name === "string" && name.length > 0;
    const valid = Array.isArray(target)
      ? target.length > 0 && target.every(isFieldName)
      : isFieldName(target);
    if (!valid) {
      check.fail(`${path}.${key}`, "must name one or more form fields");
    }
  });
}

function checkReview(check: ManifestChecker, review: unknown): void {
  if (!isRecord(review)) {
    check.fail("review", "must be an object");
    return;
  }
  check.copy(review.title, "review.title");
  check.copy(review.confirmLabel, "review.confirmLabel");
  if (review.note !== undefined) check.copy(review.note, "review.note");
  if (review.emptyValueText !== undefined) {
    check.copy(review.emptyValueText, "review.emptyValueText");
  }
  if (!isNonEmptyArray(review.summary)) {
    check.fail("review.summary", "must be a non-empty array");
    return;
  }
  review.summary.forEach((row, index) => {
    const path = `review.summary[${index}]`;
    if (!isRecord(row)) {
      check.fail(path, "must be an object");
      return;
    }
    check.copy(row.label, `${path}.label`);
    check.templateCopy(row.value, `${path}.value`);
  });
}

function checkSubmit(
  check: ManifestChecker,
  submit: unknown,
  setupMode: unknown,
): void {
  if (!isRecord(submit)) {
    check.fail("submit", "must be an object");
    return;
  }

  if (!isOneOf(submit.action, MANIFEST_ACTIONS)) {
    check.fail("submit.action", "is not an allowlisted action");
    return;
  }
  const action: ManifestAction = submit.action;

  const expectedAction: ManifestAction =
    setupMode === "assisted" ? "conversation.start" : "automation.create";
  if (action !== expectedAction) {
    check.fail(
      "submit.action",
      `does not match setupMode "${String(setupMode)}"`,
    );
    return;
  }

  if (action === "automation.create") {
    const endpoint = submit.endpoint;
    if (!isRecord(endpoint)) {
      check.fail("submit.endpoint", "must be an object");
    } else {
      if (endpoint.method !== "POST") {
        check.fail("submit.endpoint.method", "must be POST");
      }
      check.servicePath(endpoint.path, "submit.endpoint.path");
    }
    check.requestBody(submit.payload, "submit.payload");
  } else {
    if (check.templateCopy(submit.message, "submit.message")) {
      if ((submit.message as string).length > MAX_ASSISTED_MESSAGE_LENGTH) {
        check.fail(
          "submit.message",
          `must be at most ${MAX_ASSISTED_MESSAGE_LENGTH} characters`,
        );
      }
    }
  }

  const onSuccess = submit.onSuccess;
  if (!isRecord(onSuccess)) {
    check.fail("submit.onSuccess", "must be an object");
  } else {
    if (onSuccess.behavior !== "navigate") {
      check.fail("submit.onSuccess.behavior", "must be navigate");
    }
    check.inAppPath(onSuccess.to, "submit.onSuccess.to");
  }

  const onError = submit.onError;
  if (!isRecord(onError)) {
    check.fail("submit.onError", "must be an object");
    return;
  }
  if (onError.behavior !== "stayOnForm") {
    check.fail("submit.onError.behavior", "must be stayOnForm");
  }
  if (!isOneOf(onError.errorTarget, ERROR_TARGETS)) {
    check.fail("submit.onError.errorTarget", "must be field or form");
  }
  if (
    onError.reuseErrorMap !== undefined &&
    typeof onError.reuseErrorMap !== "boolean"
  ) {
    check.fail("submit.onError.reuseErrorMap", "must be a boolean");
  }
  if (onError.message !== undefined) {
    check.copy(onError.message, "submit.onError.message");
  }
}

function checkAnalytics(check: ManifestChecker, analytics: unknown): void {
  if (!isRecord(analytics)) {
    check.fail("analytics", "must be an object");
    return;
  }
  if (analytics.consent !== "required") {
    check.fail("analytics.consent", "must be required");
  }
  if (!isNonEmptyArray(analytics.stages)) {
    check.fail("analytics.stages", "must be a non-empty array");
    return;
  }
  analytics.stages.forEach((stage, index) => {
    const path = `analytics.stages[${index}]`;
    if (!isRecord(stage)) {
      check.fail(path, "must be an object");
      return;
    }
    if (typeof stage.id !== "string" || !ANALYTICS_ID_PATTERN.test(stage.id)) {
      check.fail(`${path}.id`, "must be a snake_case identifier");
    }
    if (!isOneOf(stage.on, ANALYTICS_EVENTS)) {
      check.fail(`${path}.on`, "names an event this host does not emit");
    }
    if (
      !isRecord(stage.properties) ||
      Object.keys(stage.properties).length === 0
    ) {
      check.fail(`${path}.properties`, "must be a non-empty object");
      return;
    }
    Object.entries(stage.properties).forEach(([key, value]) => {
      if (!ANALYTICS_PROPERTY_PATTERN.test(key)) {
        check.fail(`${path}.properties.${key}`, "must be a snake_case name");
      }
      if (typeof value === "string") {
        check.placeholders(value, `${path}.properties.${key}`);
      } else if (typeof value !== "number" && typeof value !== "boolean") {
        check.fail(
          `${path}.properties.${key}`,
          "must be a string, number, or boolean",
        );
      }
    });
  });
}

/**
 * Decide whether this host will act on a manifest. Returns every reason it
 * would not, so a manifest author sees the whole picture at once.
 */
export function validateManifest(candidate: unknown): ManifestValidationResult {
  const check = new ManifestChecker();

  if (!isRecord(candidate)) {
    return { valid: false, errors: ["manifest: must be an object"] };
  }

  // Version first: it selects the rules everything below is checked against, so
  // an unrecognized version must fail closed rather than be read optimistically.
  if (candidate.manifestVersion !== MANIFEST_VERSION) {
    return {
      valid: false,
      errors: [
        `manifestVersion: this host supports "${MANIFEST_VERSION}", got "${String(candidate.manifestVersion)}"`,
      ],
    };
  }

  if (typeof candidate.id !== "string" || candidate.id.length === 0) {
    check.fail("id", "must be a non-empty string");
  }
  check.copy(candidate.name, "name");
  check.copy(candidate.category, "category");
  check.copy(candidate.description, "description");
  if (!isOneOf(candidate.setupMode, ["direct", "assisted"] as const)) {
    check.fail("setupMode", "must be direct or assisted");
  }

  checkRoutes(check, candidate.routes);
  checkForm(check, candidate.form);
  checkReview(check, candidate.review);
  checkSubmit(check, candidate.submit, candidate.setupMode);
  checkAnalytics(check, candidate.analytics);

  if (candidate.capabilities !== undefined) {
    checkCapabilities(check, candidate.capabilities);
  }
  if (candidate.requires !== undefined) {
    checkRequires(check, candidate.requires);
  }
  if (candidate.validation !== undefined) {
    checkValidation(check, candidate.validation);
  } else if (candidate.setupMode === "direct") {
    // A direct setup submits a mapped payload, so it must declare how invalid
    // input is surfaced before anything is created.
    check.fail("validation", "is required for direct setup");
  }

  return { valid: check.errors.length === 0, errors: check.errors };
}

export function isValidManifest(
  candidate: unknown,
): candidate is ExtensionManifest {
  return validateManifest(candidate).valid;
}
