/**
 * Admission policy for the extension-published Automation interface manifest.
 *
 * Like a setup entry, the interface manifest is data authored in a different
 * repository that instructs this host to render copy, build links, and address
 * requests, so the host decides what it is *permitted* to state. Admission is
 * all-or-nothing: one bad field rejects the whole manifest, and the host falls
 * back to its own defaults rather than rendering a partially-trusted mix.
 */

import type {
  AutomationAttributeName,
  InterfaceAttributeType,
  InterfaceRoutes,
} from "./types";
import { INTERFACE_VERSION } from "./types";

/** Copy must never be able to inject markup into the host. */
const MARKUP_PATTERN = /<[A-Za-z/!]/;
const SLUG_PATTERN = /^[a-z0-9]+(-[a-z0-9]+)*$/;
/** The one URL the manifest may state, pinned to the product documentation. */
const DOCS_URL_PREFIX = "https://docs.openhands.dev/";
const FILE_KIND_PATTERN = /^[a-z][a-z-]*$/;
const FILENAME_SUFFIX_PATTERN = /^\.[a-z][a-z.]*json$/;
const EVENT_SOURCE_PATTERN = /^[a-z0-9][a-z0-9.-]*$/;
/** Service-relative only: rooted, no traversal, no query, no fragment. */
const ENDPOINT_PATTERN = /^\/[A-Za-z0-9/{}_-]*$/;
/** Exactly one `{id}` substitution and no other braces. */
const ID_ENDPOINT_PATTERN = /^[^{}]*\{id\}[^{}]*$/;

const GIT_PROVIDERS = ["github", "gitlab", "bitbucket"] as const;
/**
 * The attribute semantics this host's edit dialog implements: the control it
 * renders per property and the requiredness it enforces. Admission pins a
 * declared attribute to exactly these, so an admitted manifest can never
 * promise a control, a requiredness, or (below) a minimum the form would
 * silently ignore.
 */
const HOST_ATTRIBUTES: Record<
  AutomationAttributeName,
  { type: InterfaceAttributeType; required: boolean }
> = {
  name: { type: "text", required: true },
  prompt: { type: "textarea", required: false },
  model: { type: "llm-profile", required: false },
  timeout: { type: "number", required: false },
  schedule: { type: "schedule", required: false },
};
const ATTRIBUTE_NAMES = Object.keys(HOST_ATTRIBUTES);
const PLAIN_ENDPOINT_NAMES = [
  "list",
  "health",
  "capabilities",
  "validate",
  "createPrompt",
  "createPlugin",
] as const;
const ID_ENDPOINT_NAMES = ["detail", "dispatch", "runs", "tarball"] as const;

export interface InterfaceValidationContext {
  /** Ids of the published automation catalog, for the featured-list check. */
  catalogIds: ReadonlySet<string>;
  /** The routes this host has registrations for. A manifest must match them. */
  mountedRoutes: InterfaceRoutes;
}

export interface InterfaceValidationResult {
  valid: boolean;
  errors: string[];
}

type Rec = Record<string, unknown>;

function isRecord(value: unknown): value is Rec {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

class InterfaceChecker {
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

  /** A closed object: every present key must be an expected one. */
  closed(container: Rec, allowed: readonly string[], path: string): void {
    Object.keys(container)
      .filter((key) => !allowed.includes(key))
      .forEach((key) => this.fail(`${path}.${key}`, "is not allowed"));
  }

  record(value: unknown, path: string): value is Rec {
    if (isRecord(value)) return true;
    this.fail(path, "must be an object");
    return false;
  }
}

function checkRoutes(
  check: InterfaceChecker,
  routes: unknown,
  mounted: InterfaceRoutes,
): void {
  if (!check.record(routes, "routes")) return;
  check.closed(routes, ["list", "setup", "detail"], "routes");

  // The host serves what it has registrations for, so a declared route must be
  // exactly the mounted shape — the manifest owns link construction, not the
  // router table.
  (["list", "setup", "detail"] as const).forEach((name) => {
    if (routes[name] !== mounted[name]) {
      check.fail(`routes.${name}`, `must be "${mounted[name]}"`);
    }
  });
}

function checkNavigation(check: InterfaceChecker, navigation: unknown): void {
  if (!check.record(navigation, "navigation")) return;
  check.closed(navigation, ["sidebar", "commandMenu"], "navigation");

  if (check.record(navigation.sidebar, "navigation.sidebar")) {
    check.closed(navigation.sidebar, ["label"], "navigation.sidebar");
    check.copy(navigation.sidebar.label, "navigation.sidebar.label");
  }
  if (check.record(navigation.commandMenu, "navigation.commandMenu")) {
    const menu = navigation.commandMenu;
    check.closed(
      menu,
      ["title", "description", "keywords"],
      "navigation.commandMenu",
    );
    check.copy(menu.title, "navigation.commandMenu.title");
    check.copy(menu.description, "navigation.commandMenu.description");
    check.copy(menu.keywords, "navigation.commandMenu.keywords");
  }
}

function checkPages(check: InterfaceChecker, pages: unknown): void {
  if (!check.record(pages, "pages")) return;
  check.closed(pages, ["list", "detail", "edit"], "pages");

  if (check.record(pages.list, "pages.list")) {
    check.closed(pages.list, ["title", "subtitle"], "pages.list");
    check.copy(pages.list.title, "pages.list.title");
    check.copy(pages.list.subtitle, "pages.list.subtitle");
  }
  if (check.record(pages.detail, "pages.detail")) {
    check.closed(pages.detail, ["backLabel"], "pages.detail");
    check.copy(pages.detail.backLabel, "pages.detail.backLabel");
  }
  if (check.record(pages.edit, "pages.edit")) {
    check.closed(pages.edit, ["title"], "pages.edit");
    check.copy(pages.edit.title, "pages.edit.title");
  }
}

function checkAttribute(
  check: InterfaceChecker,
  host: { type: InterfaceAttributeType; required: boolean },
  attribute: unknown,
  path: string,
): void {
  if (!check.record(attribute, path)) return;
  check.closed(
    attribute,
    ["type", "label", "help", "required", "constraints"],
    path,
  );

  const { type, label, help, required, constraints } = attribute;
  if (type !== host.type) {
    check.fail(
      `${path}.type`,
      `must be "${host.type}", the control this host renders`,
    );
  }
  check.copy(label, `${path}.label`);
  if (help !== undefined) check.copy(help, `${path}.help`);
  if (required !== host.required) {
    check.fail(
      `${path}.required`,
      `must be ${host.required}, what this host enforces`,
    );
  }

  if (constraints !== undefined) {
    if (host.type !== "number") {
      check.fail(
        `${path}.constraints`,
        "is only allowed on a number attribute",
      );
    } else if (check.record(constraints, `${path}.constraints`)) {
      check.closed(constraints, ["min", "max"], `${path}.constraints`);
      const { min, max } = constraints;
      // The form validates a positive integer, i.e. an effective minimum of 1;
      // any other declared minimum would be a promise it does not keep.
      if (min !== undefined && min !== 1) {
        check.fail(
          `${path}.constraints.min`,
          "must be the 1 this host enforces",
        );
      }
      if (
        max !== undefined &&
        (!Number.isInteger(max) || (max as number) < 1)
      ) {
        check.fail(`${path}.constraints.max`, "must be a positive integer");
      }
    }
  }
}

function checkAttributes(check: InterfaceChecker, attributes: unknown): void {
  if (!check.record(attributes, "attributes")) return;
  const names = Object.keys(attributes);
  if (names.length === 0) {
    check.fail("attributes", "must declare at least one attribute");
  }
  names.forEach((name) => {
    if (!ATTRIBUTE_NAMES.includes(name)) {
      check.fail(`attributes.${name}`, "is not a settable attribute");
      return;
    }
    checkAttribute(
      check,
      HOST_ATTRIBUTES[name as AutomationAttributeName],
      attributes[name],
      `attributes.${name}`,
    );
  });
}

function checkImportExport(
  check: InterfaceChecker,
  importExport: unknown,
): void {
  if (!check.record(importExport, "importExport")) return;
  check.closed(
    importExport,
    ["fileKind", "fileVersion", "filenameSuffix", "importDefaults"],
    "importExport",
  );

  const { fileKind, fileVersion, filenameSuffix, importDefaults } =
    importExport;
  if (typeof fileKind !== "string" || !FILE_KIND_PATTERN.test(fileKind)) {
    check.fail("importExport.fileKind", "must be a lowercase slug");
  }
  if (fileVersion !== 1) {
    check.fail("importExport.fileVersion", "must be 1");
  }
  if (
    typeof filenameSuffix !== "string" ||
    !FILENAME_SUFFIX_PATTERN.test(filenameSuffix)
  ) {
    check.fail("importExport.filenameSuffix", "must be a .json suffix");
  }
  if (check.record(importDefaults, "importExport.importDefaults")) {
    check.closed(
      importDefaults,
      ["repoProvider", "placeholderEventSource"],
      "importExport.importDefaults",
    );
    if (
      typeof importDefaults.repoProvider !== "string" ||
      !(GIT_PROVIDERS as readonly string[]).includes(
        importDefaults.repoProvider,
      )
    ) {
      check.fail(
        "importExport.importDefaults.repoProvider",
        "is not a supported provider",
      );
    }
    if (
      typeof importDefaults.placeholderEventSource !== "string" ||
      !EVENT_SOURCE_PATTERN.test(importDefaults.placeholderEventSource)
    ) {
      check.fail(
        "importExport.importDefaults.placeholderEventSource",
        "must be a lowercase event source",
      );
    }
  }
}

function checkEndpoints(check: InterfaceChecker, endpoints: unknown): void {
  if (!check.record(endpoints, "endpoints")) return;
  const allowed = [...PLAIN_ENDPOINT_NAMES, ...ID_ENDPOINT_NAMES];
  check.closed(endpoints, allowed, "endpoints");

  const endpointAt = (name: string): string | null => {
    const value = endpoints[name];
    if (
      typeof value !== "string" ||
      !ENDPOINT_PATTERN.test(value) ||
      value.includes("//")
    ) {
      check.fail(`endpoints.${name}`, "must be a rooted service-relative path");
      return null;
    }
    return value;
  };

  PLAIN_ENDPOINT_NAMES.forEach((name) => {
    const value = endpointAt(name);
    if (value !== null && /[{}]/.test(value)) {
      check.fail(`endpoints.${name}`, "must not carry a substitution");
    }
  });
  ID_ENDPOINT_NAMES.forEach((name) => {
    const value = endpointAt(name);
    if (value !== null && !ID_ENDPOINT_PATTERN.test(value)) {
      check.fail(`endpoints.${name}`, "must carry exactly one {id}");
    }
  });
}

function checkSlugList(
  check: InterfaceChecker,
  value: unknown,
  path: string,
): string[] {
  if (!Array.isArray(value) || value.length === 0) {
    check.fail(path, "must be a non-empty array");
    return [];
  }
  const slugs = value.filter((item): item is string => {
    if (typeof item === "string" && SLUG_PATTERN.test(item)) return true;
    check.fail(path, "must contain only lowercase slugs");
    return false;
  });
  if (new Set(slugs).size !== slugs.length) {
    check.fail(path, "must not repeat an id");
  }
  return slugs;
}

const TOP_LEVEL_KEYS = [
  "version",
  "routes",
  "navigation",
  "pages",
  "docsUrl",
  "attributes",
  "importExport",
  "endpoints",
  "featuredAutomationIds",
  "responderIntegrationIds",
];

/**
 * Decide whether this host will act on a published interface manifest.
 *
 * The version is checked first and fails closed: a format this host does not
 * recognise is refused rather than interpreted with today's rules.
 */
export function validateInterfaceManifest(
  candidate: unknown,
  context: InterfaceValidationContext,
): InterfaceValidationResult {
  if (!isRecord(candidate)) {
    return { valid: false, errors: ["interface: must be an object"] };
  }
  if (candidate.version !== INTERFACE_VERSION) {
    return {
      valid: false,
      errors: [`interface.version: must be "${INTERFACE_VERSION}"`],
    };
  }

  const check = new InterfaceChecker();
  check.closed(candidate, TOP_LEVEL_KEYS, "interface");

  checkRoutes(check, candidate.routes, context.mountedRoutes);
  checkNavigation(check, candidate.navigation);
  checkPages(check, candidate.pages);
  if (
    typeof candidate.docsUrl !== "string" ||
    !candidate.docsUrl.startsWith(DOCS_URL_PREFIX)
  ) {
    check.fail("docsUrl", `must start with ${DOCS_URL_PREFIX}`);
  }
  checkAttributes(check, candidate.attributes);
  checkImportExport(check, candidate.importExport);
  checkEndpoints(check, candidate.endpoints);

  checkSlugList(
    check,
    candidate.featuredAutomationIds,
    "featuredAutomationIds",
  ).forEach((id) => {
    if (!context.catalogIds.has(id)) {
      check.fail("featuredAutomationIds", `${id} is not a catalog entry`);
    }
  });
  checkSlugList(
    check,
    candidate.responderIntegrationIds,
    "responderIntegrationIds",
  );

  return { valid: check.errors.length === 0, errors: check.errors };
}
