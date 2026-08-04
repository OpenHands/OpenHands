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
  OverviewMetric,
} from "./types";
import {
  DASHBOARD_FILTER_IDS,
  DASHBOARD_FILTER_VALUES,
  DASHBOARD_SORT_VALUES,
  INTERFACE_ICON_SLUGS,
  INTERFACE_SUB_PAGE_IDS,
  INTERFACE_VERSION,
  OVERVIEW_METRICS,
  OVERVIEW_TILE_PLACEHOLDERS,
} from "./types";

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

function isOneOf<T extends readonly string[]>(
  value: unknown,
  values: T,
): value is T[number] {
  return (
    typeof value === "string" && (values as readonly string[]).includes(value)
  );
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
  check.closed(routes, ["list", "setup", "detail", "templates"], "routes");

  // The host serves what it has registrations for, so a declared route must be
  // exactly the mounted shape — the manifest owns link construction, not the
  // router table.
  (["list", "setup", "detail"] as const).forEach((name) => {
    if (routes[name] !== mounted[name]) {
      check.fail(`routes.${name}`, `must be "${mounted[name]}"`);
    }
  });
  if (
    routes.templates !== undefined &&
    routes.templates !== mounted.templates
  ) {
    check.fail(
      "routes.templates",
      mounted.templates === undefined
        ? "is not a route this host mounts"
        : `must be "${mounted.templates}"`,
    );
  }
}

function checkSubPages(check: InterfaceChecker, subPages: unknown): void {
  if (!Array.isArray(subPages) || subPages.length === 0) {
    check.fail("navigation.subPages", "must be a non-empty array");
    return;
  }
  const seen = new Set<string>();
  subPages.forEach((item: unknown, index) => {
    const path = `navigation.subPages[${index}]`;
    if (!check.record(item, path)) return;
    check.closed(item, ["page", "label", "icon"], path);

    const { page, label, icon } = item;
    if (!isOneOf(page, INTERFACE_SUB_PAGE_IDS)) {
      check.fail(`${path}.page`, "is not a sub-page this host serves");
    } else if (seen.has(page)) {
      check.fail(`${path}.page`, "repeats a page");
    } else {
      seen.add(page);
    }
    check.copy(label, `${path}.label`);
    checkIconSlug(check, icon, `${path}.icon`);
  });
}

function checkIconSlug(
  check: InterfaceChecker,
  icon: unknown,
  path: string,
): void {
  if (!isOneOf(icon, INTERFACE_ICON_SLUGS)) {
    check.fail(path, "is not an icon this host ships");
  }
}

function checkNavigation(check: InterfaceChecker, navigation: unknown): void {
  if (!check.record(navigation, "navigation")) return;
  check.closed(
    navigation,
    ["sidebar", "commandMenu", "subPages"],
    "navigation",
  );

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
  if (navigation.subPages !== undefined) {
    checkSubPages(check, navigation.subPages);
  }
}

function checkPages(check: InterfaceChecker, pages: unknown): void {
  if (!check.record(pages, "pages")) return;
  check.closed(pages, ["list", "detail", "edit", "templates"], "pages");

  if (check.record(pages.list, "pages.list")) {
    check.closed(
      pages.list,
      ["title", "subtitle", "overview", "filters", "sort", "insights"],
      "pages.list",
    );
    check.copy(pages.list.title, "pages.list.title");
    check.copy(pages.list.subtitle, "pages.list.subtitle");
    if (pages.list.overview !== undefined) {
      checkOverview(check, pages.list.overview);
    }
    if (pages.list.filters !== undefined) {
      checkFilters(check, pages.list.filters);
    }
    if (pages.list.sort !== undefined) {
      checkSort(check, pages.list.sort);
    }
    if (pages.list.insights !== undefined) {
      checkInsights(check, pages.list.insights);
    }
  }
  if (check.record(pages.detail, "pages.detail")) {
    check.closed(pages.detail, ["backLabel"], "pages.detail");
    check.copy(pages.detail.backLabel, "pages.detail.backLabel");
  }
  if (check.record(pages.edit, "pages.edit")) {
    check.closed(pages.edit, ["title"], "pages.edit");
    check.copy(pages.edit.title, "pages.edit.title");
  }
  if (pages.templates !== undefined) {
    checkTemplatesPage(check, pages.templates);
  }
}

function checkOverview(check: InterfaceChecker, overview: unknown): void {
  const path = "pages.list.overview";
  if (!check.record(overview, path)) return;
  check.closed(overview, ["label", "tiles"], path);
  check.copy(overview.label, `${path}.label`);

  const { tiles } = overview;
  if (!Array.isArray(tiles) || tiles.length === 0) {
    check.fail(`${path}.tiles`, "must be a non-empty array");
    return;
  }
  const seen = new Set<string>();
  tiles.forEach((tile: unknown, index) => {
    checkOverviewTile(check, tile, seen, `${path}.tiles[${index}]`);
  });
}

function checkOverviewTile(
  check: InterfaceChecker,
  tile: unknown,
  seen: Set<string>,
  path: string,
): void {
  if (!check.record(tile, path)) return;
  check.closed(tile, ["metric", "label", "detail", "zeroDetail", "icon"], path);

  const { metric, label, detail, zeroDetail, icon } = tile;
  check.copy(label, `${path}.label`);
  checkIconSlug(check, icon, `${path}.icon`);
  if (!isOneOf(metric, OVERVIEW_METRICS)) {
    check.fail(`${path}.metric`, "is not a metric this host computes");
    check.copy(detail, `${path}.detail`);
    if (zeroDetail !== undefined) check.copy(zeroDetail, `${path}.zeroDetail`);
    return;
  }
  if (seen.has(metric)) {
    check.fail(`${path}.metric`, "repeats a metric");
  } else {
    seen.add(metric);
  }
  checkTileCopy(check, detail, metric, `${path}.detail`);
  if (zeroDetail !== undefined) {
    checkTileCopy(check, zeroDetail, metric, `${path}.zeroDetail`);
  }
}

/**
 * Tile copy may embed `{{name}}` placeholders, but only the names the metric
 * exposes — plain substitution, never an expression.
 */
function checkTileCopy(
  check: InterfaceChecker,
  value: unknown,
  metric: OverviewMetric,
  path: string,
): void {
  if (!check.copy(value, path)) return;
  const exposed = OVERVIEW_TILE_PLACEHOLDERS[metric];
  const unknownPlaceholder =
    exposed.length === 0
      ? /\{\{/
      : new RegExp(`\\{\\{(?!(?:${exposed.join("|")})\\}\\})`);
  if (unknownPlaceholder.test(value as string)) {
    check.fail(path, "uses a placeholder this metric does not expose");
  }
}

function checkFilters(check: InterfaceChecker, filters: unknown): void {
  const path = "pages.list.filters";
  if (!Array.isArray(filters) || filters.length === 0) {
    check.fail(path, "must be a non-empty array");
    return;
  }
  const seen = new Set<string>();
  filters.forEach((filter: unknown, index) => {
    checkFilter(check, filter, seen, `${path}[${index}]`);
  });
}

function checkFilter(
  check: InterfaceChecker,
  filter: unknown,
  seen: Set<string>,
  path: string,
): void {
  if (!check.record(filter, path)) return;
  check.closed(filter, ["id", "label", "options"], path);

  const { id, label, options } = filter;
  check.copy(label, `${path}.label`);
  if (!isOneOf(id, DASHBOARD_FILTER_IDS)) {
    check.fail(`${path}.id`, "is not a filter this host implements");
    return;
  }
  if (seen.has(id)) {
    check.fail(`${path}.id`, "repeats a filter");
  } else {
    seen.add(id);
  }
  checkFilterOptions(
    check,
    options,
    DASHBOARD_FILTER_VALUES[id],
    `${path}.options`,
  );
}

function checkFilterOptions(
  check: InterfaceChecker,
  options: unknown,
  values: readonly string[],
  path: string,
): void {
  if (!Array.isArray(options) || options.length < 2) {
    check.fail(path, "must offer at least two options");
    return;
  }
  const seen = new Set<string>();
  options.forEach((option: unknown, index) => {
    const optionPath = `${path}[${index}]`;
    if (!check.record(option, optionPath)) return;
    check.closed(option, ["value", "label"], optionPath);

    const { value, label } = option;
    if (!isOneOf(value, values)) {
      check.fail(`${optionPath}.value`, "is not a value this host implements");
    } else if (seen.has(value)) {
      check.fail(`${optionPath}.value`, "repeats a value");
    } else {
      seen.add(value);
    }
    check.copy(label, `${optionPath}.label`);
  });
  // "all" is the host's initial selection and what Clear filters resets to,
  // so a filter that does not offer it could never be neutral.
  if (!seen.has("all")) {
    check.fail(path, 'must offer the "all" option');
  }
}

function checkSort(check: InterfaceChecker, sort: unknown): void {
  const path = "pages.list.sort";
  if (!check.record(sort, path)) return;
  check.closed(sort, ["label", "options", "default"], path);

  const { label, options, default: defaultValue } = sort;
  check.copy(label, `${path}.label`);
  const declared = new Set<string>();
  if (!Array.isArray(options) || options.length === 0) {
    check.fail(`${path}.options`, "must be a non-empty array");
  } else {
    options.forEach((option: unknown, index) => {
      const optionPath = `${path}.options[${index}]`;
      if (!check.record(option, optionPath)) return;
      check.closed(option, ["value", "label"], optionPath);

      if (!isOneOf(option.value, DASHBOARD_SORT_VALUES)) {
        check.fail(`${optionPath}.value`, "is not a sort this host implements");
      } else if (declared.has(option.value)) {
        check.fail(`${optionPath}.value`, "repeats a value");
      } else {
        declared.add(option.value);
      }
      check.copy(option.label, `${optionPath}.label`);
    });
  }
  if (typeof defaultValue !== "string" || !declared.has(defaultValue)) {
    check.fail(`${path}.default`, "must be one of the declared option values");
  }
}

/** Every insight state and stat the host renders must have a caption. */
const INSIGHTS_SHAPE = {
  health: ["healthy", "failing", "running", "disabled", "neverRun", "checking"],
  lastRun: ["label", "never", "justNow"],
  stats: ["runs", "recentSuccess", "averageDuration"],
} as const;

function checkInsights(check: InterfaceChecker, insights: unknown): void {
  const path = "pages.list.insights";
  if (!check.record(insights, path)) return;
  check.closed(insights, Object.keys(INSIGHTS_SHAPE), path);

  (Object.keys(INSIGHTS_SHAPE) as (keyof typeof INSIGHTS_SHAPE)[]).forEach(
    (section) => {
      const sectionPath = `${path}.${section}`;
      const value = insights[section];
      if (!check.record(value, sectionPath)) return;
      const keys = INSIGHTS_SHAPE[section];
      check.closed(value, keys, sectionPath);
      keys.forEach((key) => {
        check.copy(value[key], `${sectionPath}.${key}`);
      });
    },
  );
}

function checkTemplatesPage(check: InterfaceChecker, templates: unknown): void {
  const path = "pages.templates";
  if (!check.record(templates, path)) return;
  check.closed(templates, ["title", "description"], path);
  check.copy(templates.title, `${path}.title`);
  check.copy(templates.description, `${path}.description`);
}

/**
 * The sub-page surface is declared whole or not at all. Navigation that points
 * at an unrouted page, or a dashboard with filters but no insight captions,
 * would be a partially-trusted mix, and the host refuses those wholesale.
 */
function checkSubPageGroup(check: InterfaceChecker, candidate: Rec): void {
  const routes = isRecord(candidate.routes) ? candidate.routes : {};
  const navigation = isRecord(candidate.navigation) ? candidate.navigation : {};
  const pages = isRecord(candidate.pages) ? candidate.pages : {};
  const list = isRecord(pages.list) ? pages.list : {};

  const pieces: [string, unknown][] = [
    ["routes.templates", routes.templates],
    ["navigation.subPages", navigation.subPages],
    ["pages.templates", pages.templates],
    ["pages.list.overview", list.overview],
    ["pages.list.filters", list.filters],
    ["pages.list.sort", list.sort],
    ["pages.list.insights", list.insights],
  ];
  const missing = pieces
    .filter(([, value]) => value === undefined)
    .map(([name]) => name);
  if (missing.length === 0 || missing.length === pieces.length) return;
  check.fail(
    "interface",
    `the sub-page surface must be declared whole; missing ${missing.join(", ")}`,
  );
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

  checkSubPageGroup(check, candidate);

  return { valid: check.errors.length === 0, errors: check.errors };
}
