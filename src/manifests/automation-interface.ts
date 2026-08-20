/**
 * The automation interface seam.
 *
 * Every automation-specific datum the host's surfaces render — navigation and
 * page copy, the settable attributes, the import/export envelope, endpoint
 * paths, the featured and responder id lists, the sub-page surface — is served
 * from the interface manifest the pinned `@openhands/extensions` publishes.
 * The host holds none of it. When the package publishes no manifest, or the
 * published one fails admission, there is nothing to serve: the nav entries do
 * not render and the routes 404. `hasAutomationInterface()` is that gate, and
 * every accessor that needs the manifest is reachable only from behind it.
 *
 * Routes are the exception, and they are not a definition: `src/routes.ts`
 * mounts them, so the host states them here and admission checks a manifest's
 * routes against them. That check is what keeps a published deep link
 * resolving to the page it named.
 */

import { AUTOMATION_CATALOG } from "@openhands/extensions/automations";
import { validateInterfaceManifest } from "./interface-validation";
import { AUTOMATION_INTERFACE_CANDIDATE } from "./manifest-sources";
import type {
  AutomationAttributeName,
  InterfaceDashboardFilter,
  InterfaceDashboardSort,
  InterfaceEndpointName,
  InterfaceIconSlug,
  InterfaceImportExport,
  InterfaceListInsights,
  InterfaceManifest,
  InterfaceOverview,
  InterfaceRoutes,
  InterfaceSubPageId,
  InterfaceTemplatesPage,
} from "./types";

/** The routes this host has registrations for, in `src/routes.ts`. */
const MOUNTED_ROUTES = {
  list: "/automations",
  setup: "/automations/new/:automationId",
  detail: "/automations/:automationId",
  templates: "/automations/templates",
} satisfies InterfaceRoutes;

/**
 * File-format section for portable `.automation.json` export/import. A docs
 * pointer for the host's import picker, not interface data — the manifest's
 * `importExport` block has no docs field.
 */
export const AUTOMATION_FILE_FORMAT_DOCS_URL =
  "https://docs.openhands.dev/openhands/usage/agent-canvas/managing-automations#exported-file-format";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function getCatalogIds(): ReadonlySet<string> {
  return new Set(
    (AUTOMATION_CATALOG as readonly unknown[]).flatMap((entry) =>
      isRecord(entry) && typeof entry.id === "string" ? [entry.id] : [],
    ),
  );
}

function admitInterfaceManifest(candidate: unknown): InterfaceManifest | null {
  if (candidate === undefined) return null;

  const result = validateInterfaceManifest(candidate, {
    catalogIds: getCatalogIds(),
    mountedRoutes: MOUNTED_ROUTES,
  });
  if (!result.valid) {
    // Mirrors the setup registry: a rejected manifest is skipped loudly, and
    // the surfaces it would have described stay unrendered.
    console.warn(
      "Rejected the automation interface manifest:",
      result.errors.join("; "),
    );
    return null;
  }
  return candidate as InterfaceManifest;
}

const ADMITTED = admitInterfaceManifest(AUTOMATION_INTERFACE_CANDIDATE);

/**
 * Whether this deployment has an automation interface at all. The nav entries
 * and the route loaders ask before rendering; nothing else has to.
 */
export function hasAutomationInterface(): boolean {
  return ADMITTED !== null;
}

/**
 * The admitted manifest, for the accessors that cannot answer without one.
 * Reaching one of those without a manifest means a surface rendered past its
 * gate — a wiring mistake, not a state to render.
 */
function requireInterface(): InterfaceManifest {
  if (!ADMITTED) {
    throw new Error(
      "No automation interface manifest is admitted, so this surface should not have rendered.",
    );
  }
  return ADMITTED;
}

function substituteRouteParam(pattern: string, id: string): string {
  return pattern.replace(":automationId", encodeURIComponent(id));
}

export function automationListPath(): string {
  return MOUNTED_ROUTES.list;
}

export function automationSetupPath(id: string): string {
  return substituteRouteParam(MOUNTED_ROUTES.setup, id);
}

export function automationDetailPath(id: string): string {
  return substituteRouteParam(MOUNTED_ROUTES.detail, id);
}

export function automationTemplatesPath(): string {
  return MOUNTED_ROUTES.templates;
}

/**
 * A declared endpoint path. Empty for one the manifest may omit - the two a
 * bundle needs were added after the block shipped - so a caller that needs one
 * says so rather than reading a host-held default that does not exist.
 */
export function getAutomationEndpoint(name: InterfaceEndpointName): string {
  return requireInterface().endpoints[name] ?? "";
}

/** An id-parameterized endpoint with `{id}` substituted, encoded. */
export function getAutomationIdEndpoint(
  name: "detail" | "dispatch" | "runs" | "tarball",
  id: string,
): string {
  return getAutomationEndpoint(name).replace("{id}", encodeURIComponent(id));
}

export interface InterfaceCopy {
  sidebarLabel: string;
  commandMenuTitle: string;
  commandMenuDescription: string;
  commandMenuKeywords: string;
  listTitle: string;
  listSubtitle: string;
  detailBackLabel: string;
  editTitle: string;
}

export function getInterfaceCopy(): InterfaceCopy {
  const manifest = requireInterface();
  return {
    sidebarLabel: manifest.navigation.sidebar.label,
    commandMenuTitle: manifest.navigation.commandMenu.title,
    commandMenuDescription: manifest.navigation.commandMenu.description,
    commandMenuKeywords: manifest.navigation.commandMenu.keywords,
    listTitle: manifest.pages.list.title,
    listSubtitle: manifest.pages.list.subtitle,
    detailBackLabel: manifest.pages.detail.backLabel,
    editTitle: manifest.pages.edit.title,
  };
}

export interface AttributeSpec {
  present: boolean;
  label: string;
  /** Null when the manifest states no help text for the attribute. */
  help: string | null;
  required: boolean;
  min: number | null;
  max: number | null;
}

/** An attribute the manifest does not declare is not offered at all. */
const ABSENT_ATTRIBUTE: AttributeSpec = {
  present: false,
  label: "",
  help: null,
  required: false,
  min: null,
  max: null,
};

export function getAttributeSpec(name: AutomationAttributeName): AttributeSpec {
  const attribute = requireInterface().attributes[name];
  if (!attribute) return ABSENT_ATTRIBUTE;
  return {
    present: true,
    label: attribute.label,
    help: attribute.help ?? null,
    required: attribute.required,
    min: attribute.constraints?.min ?? null,
    max: attribute.constraints?.max ?? null,
  };
}

export function getImportExportSpec(): InterfaceImportExport {
  return requireInterface().importExport;
}

export function getAutomationsDocsUrl(): string {
  return requireInterface().docsUrl;
}

export function getFeaturedAutomationIds(): readonly string[] {
  return requireInterface().featuredAutomationIds;
}

export function getResponderIntegrationIds(): readonly string[] {
  return requireInterface().responderIntegrationIds;
}

export interface SubPageNavSpec {
  page: InterfaceSubPageId;
  /** The page's route, resolved through the host's route table. */
  to: string;
  label: string;
  icon: InterfaceIconSlug;
}

/**
 * The sub-page navigation, or null when the manifest does not declare the
 * sub-page surface.
 */
export function getSubPagesSpec(): SubPageNavSpec[] | null {
  const subPages = ADMITTED?.navigation.subPages;
  if (!subPages) return null;
  return subPages.map((item) => ({
    page: item.page,
    to:
      item.page === "templates"
        ? automationTemplatesPath()
        : automationListPath(),
    label: item.label,
    icon: item.icon,
  }));
}

export interface DashboardSpec {
  overview: InterfaceOverview;
  filters: InterfaceDashboardFilter[];
  sort: InterfaceDashboardSort;
  insights: InterfaceListInsights;
}

/**
 * The list page's dashboard composition, or null when the manifest does not
 * declare it. Admission accepts the sub-page surface whole or not at all, so
 * these four are present together.
 */
export function getDashboardSpec(): DashboardSpec | null {
  const list = ADMITTED?.pages.list;
  if (!list?.overview || !list.filters || !list.sort || !list.insights) {
    return null;
  }
  return {
    overview: list.overview,
    filters: list.filters,
    sort: list.sort,
    insights: list.insights,
  };
}

/** The templates page identity, or null when the manifest does not declare it. */
export function getTemplatesPageSpec(): InterfaceTemplatesPage | null {
  return ADMITTED?.pages.templates ?? null;
}
