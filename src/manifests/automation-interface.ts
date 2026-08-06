/**
 * The automation interface seam.
 *
 * Every automation-specific datum the host's surfaces render — routes,
 * navigation and page copy, the settable attributes, the import/export
 * envelope, endpoint paths, the featured and responder id lists — is served
 * from here.
 * The published interface manifest supplies it when the pinned
 * `@openhands/extensions` ships one and it passes admission; otherwise the
 * host's own defaults do, which reproduce today's behavior exactly. Copy
 * defaults are null, meaning "render the host's own translation".
 *
 * The sub-page surface (navigation, overview tiles, filters, sort, run
 * insights, templates page) is the exception: the host holds no definitions
 * of its own, so its accessors return null until an admitted manifest
 * declares it, and the sub-pages simply do not render.
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

const DEFAULT_ENDPOINTS: InterfaceManifest["endpoints"] = {
  list: "/v1",
  detail: "/v1/{id}",
  dispatch: "/v1/{id}/dispatch",
  runs: "/v1/{id}/runs",
  tarball: "/v1/{id}/tarball",
  health: "/health",
  capabilities: "/v1/capabilities",
  validate: "/v1/validate",
  createPrompt: "/v1/preset/prompt",
  createPlugin: "/v1/preset/plugin",
};

const DEFAULT_IMPORT_EXPORT: InterfaceImportExport = {
  fileKind: "automation",
  fileVersion: 1,
  filenameSuffix: ".automation.json",
  importDefaults: {
    repoProvider: "github",
    placeholderEventSource: "agent-canvas-import",
  },
};

const DEFAULT_DOCS_URL =
  "https://docs.openhands.dev/openhands/usage/automations/overview";

/**
 * Proven automations featured above the Beta group. NOT derived from
 * popularityRank (slack-standup-digest@94 outranks slack-channel-monitor@92
 * yet is Beta).
 */
const DEFAULT_FEATURED_AUTOMATION_IDS: readonly string[] = [
  "github-pr-reviewer",
  "github-repo-monitor",
  "slack-channel-monitor",
];

/**
 * Integrations whose automations are treated as event responders that poll
 * continuously, and so get the deployment-choice dialog.
 */
const DEFAULT_RESPONDER_INTEGRATION_IDS: readonly string[] = [
  "github",
  "slack",
];

export interface AttributeSpec {
  present: boolean;
  /** Null means "render the host's own translation". */
  label: string | null;
  help: string | null;
  required: boolean;
  min: number | null;
  max: number | null;
}

const ABSENT_ATTRIBUTE: AttributeSpec = {
  present: false,
  label: null,
  help: null,
  required: false,
  min: null,
  max: null,
};

/** Today's edit dialog, restated as specs so absence of a manifest changes nothing. */
const DEFAULT_ATTRIBUTES: Record<AutomationAttributeName, AttributeSpec> = {
  name: {
    present: true,
    label: null,
    help: null,
    required: true,
    min: null,
    max: null,
  },
  prompt: {
    present: true,
    label: null,
    help: null,
    required: false,
    min: null,
    max: null,
  },
  model: {
    present: true,
    label: null,
    help: null,
    required: false,
    min: null,
    max: null,
  },
  timeout: {
    present: true,
    label: null,
    help: null,
    required: false,
    min: 1,
    max: null,
  },
  schedule: {
    present: true,
    label: null,
    help: null,
    required: false,
    min: null,
    max: null,
  },
};

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
    // the host's defaults stand.
    console.warn(
      "Rejected the automation interface manifest:",
      result.errors.join("; "),
    );
    return null;
  }
  return candidate as InterfaceManifest;
}

const ADMITTED = admitInterfaceManifest(AUTOMATION_INTERFACE_CANDIDATE);

function substituteRouteParam(pattern: string, id: string): string {
  return pattern.replace(":automationId", encodeURIComponent(id));
}

export function automationListPath(): string {
  return (ADMITTED?.routes ?? MOUNTED_ROUTES).list;
}

export function automationSetupPath(id: string): string {
  return substituteRouteParam((ADMITTED?.routes ?? MOUNTED_ROUTES).setup, id);
}

export function automationDetailPath(id: string): string {
  return substituteRouteParam((ADMITTED?.routes ?? MOUNTED_ROUTES).detail, id);
}

export function automationTemplatesPath(): string {
  return ADMITTED?.routes.templates ?? MOUNTED_ROUTES.templates;
}

export function getAutomationEndpoint(name: InterfaceEndpointName): string {
  return (ADMITTED?.endpoints ?? DEFAULT_ENDPOINTS)[name];
}

/** An id-parameterized endpoint with `{id}` substituted, encoded. */
export function getAutomationIdEndpoint(
  name: "detail" | "dispatch" | "runs" | "tarball",
  id: string,
): string {
  return getAutomationEndpoint(name).replace("{id}", encodeURIComponent(id));
}

export interface InterfaceCopy {
  sidebarLabel: string | null;
  commandMenuTitle: string | null;
  commandMenuDescription: string | null;
  commandMenuKeywords: string | null;
  listTitle: string | null;
  listSubtitle: string | null;
  detailBackLabel: string | null;
  editTitle: string | null;
}

const HOST_COPY: InterfaceCopy = {
  sidebarLabel: null,
  commandMenuTitle: null,
  commandMenuDescription: null,
  commandMenuKeywords: null,
  listTitle: null,
  listSubtitle: null,
  detailBackLabel: null,
  editTitle: null,
};

export function getInterfaceCopy(): InterfaceCopy {
  if (!ADMITTED) return HOST_COPY;
  return {
    sidebarLabel: ADMITTED.navigation.sidebar.label,
    commandMenuTitle: ADMITTED.navigation.commandMenu.title,
    commandMenuDescription: ADMITTED.navigation.commandMenu.description,
    commandMenuKeywords: ADMITTED.navigation.commandMenu.keywords,
    listTitle: ADMITTED.pages.list.title,
    listSubtitle: ADMITTED.pages.list.subtitle,
    detailBackLabel: ADMITTED.pages.detail.backLabel,
    editTitle: ADMITTED.pages.edit.title,
  };
}

export function getAttributeSpec(name: AutomationAttributeName): AttributeSpec {
  if (!ADMITTED) return DEFAULT_ATTRIBUTES[name];

  const attribute = ADMITTED.attributes[name];
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
  return ADMITTED?.importExport ?? DEFAULT_IMPORT_EXPORT;
}

export function getAutomationsDocsUrl(): string {
  return ADMITTED?.docsUrl ?? DEFAULT_DOCS_URL;
}

export function getFeaturedAutomationIds(): readonly string[] {
  return ADMITTED?.featuredAutomationIds ?? DEFAULT_FEATURED_AUTOMATION_IDS;
}

export function getResponderIntegrationIds(): readonly string[] {
  return ADMITTED?.responderIntegrationIds ?? DEFAULT_RESPONDER_INTEGRATION_IDS;
}

export interface SubPageNavSpec {
  page: InterfaceSubPageId;
  /** The page's route, resolved through the manifest. */
  to: string;
  label: string;
  icon: InterfaceIconSlug;
}

/**
 * The sub-page navigation, or null when the manifest does not declare the
 * sub-page surface. No manifest, no sub-pages — there is no host default.
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
