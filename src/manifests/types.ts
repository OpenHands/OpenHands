/**
 * Host-owned shape of an extension-authored setup experience.
 *
 * A catalog entry is declarative data published by `@openhands/extensions`. Its
 * optional `setup` block says what to ask a user for; everything derivable from
 * that — the request to send, the route, the review screen, the analytics — is
 * this host's to generate, so none of it appears here.
 *
 * The names mirror `@openhands/extensions/automations`' own declarations, so an
 * entry from the published package assigns without an adapter.
 */

export const SETUP_VERSION = "1.0";

export type SetupMode = "direct" | "assisted";

export type SetupFieldType =
  | "text"
  | "textarea"
  | "select"
  | "cron"
  | "timezone"
  | "repo-picker";

export type SetupGitProvider = "github" | "gitlab" | "bitbucket";

export type SetupTriggerKind = "cron" | "event";

/** Placeholder namespaces a setup block may reference inside `{{...}}`. */
export const SETUP_PLACEHOLDER_NAMESPACES = ["form", "automation"] as const;

export interface SetupFieldOption {
  value: string;
  label: string;
}

export interface SetupFieldConstraints {
  minLength?: number;
  maxLength?: number;
  /**
   * A host-implemented check named from a closed set. Entries supply no regex
   * of their own, so they cannot hand the host a pathological pattern.
   */
  format?: "safeExpressionLiteral";
}

export interface SetupFormField {
  type: SetupFieldType;
  label: string;
  help: string;
  placeholder?: string;
  default?: string;
  required: boolean;
  provider?: SetupGitProvider;
  options?: SetupFieldOption[];
  constraints?: SetupFieldConstraints;
}

/** Keyed by field name, which is what `{{form.<name>}}` resolves against. */
export type SetupFormFields = Record<string, SetupFormField>;

export interface SetupForm {
  note?: string;
  /** Inputs that decide when the automation runs, keyed by trigger kind. */
  triggers?: Partial<Record<SetupTriggerKind, SetupFormFields>>;
  /** Every other input: the arguments to the automation itself. */
  args: SetupFormFields;
}

export interface SetupBlock {
  version: typeof SETUP_VERSION;
  mode: SetupMode;
  form: SetupForm;
  /** direct only. What the automation is told to do. */
  prompt?: string;
  /** direct only, event trigger only. Which delivered events belong to it. */
  filter?: string;
  /**
   * Setup context for the conversation that finishes setup. Required for
   * assisted mode. Optional for direct mode, where it seeds the fallback
   * conversation offered when the deployment cannot run the direct path.
   */
  message?: string;
}

export interface SetupIntegrationRequirement {
  /** Why this entry needs the integration. */
  message: string;
  /** Defaults to true. `false` lets setup continue while it is unconnected. */
  required?: false;
}

export interface SetupPrerequisites {
  /** Keyed by integration catalog id. */
  integrations: Record<string, SetupIntegrationRequirement>;
  /** Deployment capabilities this entry cannot run without. */
  features?: string[];
}

/**
 * The part of a catalog entry this host reads. An admitted entry always has a
 * `setup` block; entries without one are simply not setup entries.
 */
export interface SetupEntry {
  id: string;
  name: string;
  description: string;
  requires: SetupPrerequisites;
  /** The skill that owns the launch command. Defaults to `id`. */
  skill?: string;
  setup: SetupBlock;
}

export type SetupPayloadValue =
  | string
  | number
  | boolean
  | null
  | SetupPayloadValue[]
  | { [key: string]: SetupPayloadValue };

export interface SetupRequestBody {
  [key: string]: SetupPayloadValue;
}

/** Form values are collected as strings; the payload mapping shapes them. */
export type SetupFormValues = Record<string, string>;

/** `GET /v1/capabilities` — what this deployment supports. */
export interface DeploymentCapabilities {
  ready: boolean;
  /** Absolute timeout ceiling enforced by this deployment's automation API. */
  maxAutomationTimeoutSeconds?: number;
  triggerKinds: string[];
  eventSources: string[];
  eventTypes: string[];
  triggers: {
    cron?: { minIntervalSeconds: number; timezones: string[] };
    event?: { filterLanguage: string; filterFunctions: string[] };
  };
  features: string[];
}

/** A single problem with a draft, addressed to the field that caused it. */
export interface DraftValidationError {
  /** Dotted path into the draft. Null when the problem spans the whole draft. */
  field: string | null;
  code: string;
  message: string;
}

/** `POST /v1/validate` — an invalid draft is still a 200. */
export interface ValidateDraftResponse {
  valid: boolean;
  errors: DraftValidationError[];
  sampleEventMatched?: boolean | null;
}

/**
 * Host-owned shape of the production Automation interface manifest, published
 * by `@openhands/extensions/automations` as `AUTOMATION_INTERFACE`.
 *
 * The catalog states what varies per automation; this states the domain-level
 * facts of the interface itself. As with setup entries, the published data is
 * validated at admission rather than trusted, and a package that predates the
 * manifest simply leaves the host on its own defaults.
 */

export const INTERFACE_VERSION = "1.0";

/** The closed set of runtime-model properties a client may offer for setting. */
export type AutomationAttributeName =
  | "name"
  | "prompt"
  | "model"
  | "timeout"
  | "schedule";

export type InterfaceAttributeType =
  | "text"
  | "textarea"
  | "number"
  | "llm-profile"
  | "schedule";

/** How one settable attribute of an existing Automation is offered. */
export interface InterfaceAttribute {
  type: InterfaceAttributeType;
  label: string;
  help?: string;
  required: boolean;
  /** Only a `number` attribute carries constraints. */
  constraints?: { min?: number; max?: number };
}

export interface InterfaceRoutes {
  list: string;
  /** Carries the `:automationId` segment the host substitutes. */
  setup: string;
  /** Carries the `:automationId` segment the host substitutes. */
  detail: string;
  /** The templates sub-page. Static: there is no parameter to substitute. */
  templates?: string;
}

/**
 * Service-relative paths the host calls. The base path, methods, headers, and
 * auth remain the host's. `{id}` marks where the automation id is substituted.
 */
export interface InterfaceEndpoints {
  list: string;
  detail: string;
  dispatch: string;
  runs: string;
  tarball: string;
  health: string;
  capabilities: string;
  validate: string;
  createPrompt: string;
  createPlugin: string;
}

export type InterfaceEndpointName = keyof InterfaceEndpoints;

export interface InterfaceImportExport {
  fileKind: string;
  fileVersion: number;
  filenameSuffix: string;
  importDefaults: {
    /** Provider inferred for short owner/repo repository URLs on import. */
    repoProvider: SetupGitProvider;
    /** Event source of the placeholder trigger that keeps an import inert. */
    placeholderEventSource: string;
  };
}

/**
 * The sub-page surface: navigation, overview tiles, filters, sort, and run
 * insights for the list page, and a templates page. Every value below names
 * something this host implements from a closed set — a metric it computes, a
 * predicate it applies, a comparator it runs, an icon it ships — and the
 * manifest picks and captions it, never defines it. The host carries no
 * definitions of its own for this surface, so a manifest without it simply
 * leaves the sub-pages unrendered.
 */

export const INTERFACE_SUB_PAGE_IDS = ["list", "templates"] as const;

export type InterfaceSubPageId = (typeof INTERFACE_SUB_PAGE_IDS)[number];

/** Icon names this host maps to artwork. A manifest cannot supply its own. */
export const INTERFACE_ICON_SLUGS = [
  "layout-dashboard",
  "sparkles",
  "bot",
  "circle-alert",
  "activity",
  "timer",
] as const;

export type InterfaceIconSlug = (typeof INTERFACE_ICON_SLUGS)[number];

export interface InterfaceSubPageNavItem {
  /** The `pages` entry this item navigates to; its route comes from `routes`. */
  page: InterfaceSubPageId;
  label: string;
  icon: InterfaceIconSlug;
}

/** Values this host computes over the loaded automations and their runs. */
export const OVERVIEW_METRICS = [
  "automations",
  "needs-attention",
  "total-runs",
  "average-duration",
] as const;

export type OverviewMetric = (typeof OVERVIEW_METRICS)[number];

/**
 * Placeholder names a tile's `detail`/`zeroDetail` copy may reference inside
 * `{{...}}`, per metric. Plain substitution, like setup copy.
 */
export const OVERVIEW_TILE_PLACEHOLDERS: Record<
  OverviewMetric,
  readonly string[]
> = {
  automations: ["active"],
  "needs-attention": [],
  "total-runs": [],
  "average-duration": [],
};

export interface InterfaceOverviewTile {
  metric: OverviewMetric;
  label: string;
  /** Caption under the value. */
  detail: string;
  /** Replaces `detail` while the metric's value is zero. */
  zeroDetail?: string;
  icon: InterfaceIconSlug;
}

export interface InterfaceOverview {
  /** Names the tiles section for assistive technology. */
  label: string;
  tiles: InterfaceOverviewTile[];
}

/** Filter values name predicates this host implements, per filter id. */
export const DASHBOARD_FILTER_VALUES = {
  status: ["all", "active", "failing", "disabled"],
  trigger: ["all", "schedule", "event"],
} as const;

export type DashboardFilterId = keyof typeof DASHBOARD_FILTER_VALUES;

export const DASHBOARD_FILTER_IDS = Object.keys(
  DASHBOARD_FILTER_VALUES,
) as readonly DashboardFilterId[];

export type DashboardStatusValue =
  (typeof DASHBOARD_FILTER_VALUES.status)[number];

export type DashboardTriggerValue =
  (typeof DASHBOARD_FILTER_VALUES.trigger)[number];

export interface InterfaceStatusFilter {
  id: "status";
  /** The control's accessible name. */
  label: string;
  options: { value: DashboardStatusValue; label: string }[];
}

export interface InterfaceTriggerFilter {
  id: "trigger";
  /** The control's accessible name. */
  label: string;
  options: { value: DashboardTriggerValue; label: string }[];
}

export type InterfaceDashboardFilter =
  | InterfaceStatusFilter
  | InterfaceTriggerFilter;

/** Sort values name comparators this host implements. */
export const DASHBOARD_SORT_VALUES = ["last-run", "runs", "name"] as const;

export type DashboardSortValue = (typeof DASHBOARD_SORT_VALUES)[number];

export interface InterfaceDashboardSort {
  /** The control's accessible name. */
  label: string;
  options: { value: DashboardSortValue; label: string }[];
  /** One of the declared option values. */
  default: DashboardSortValue;
}

/**
 * Copy for the per-automation run insights on cards and rows. The states,
 * precedence, sampling, and value formatting are this host's; the manifest
 * names them.
 */
export interface InterfaceListInsights {
  health: {
    healthy: string;
    failing: string;
    running: string;
    disabled: string;
    neverRun: string;
    checking: string;
  };
  lastRun: { label: string; never: string; justNow: string };
  stats: { runs: string; recentSuccess: string; averageDuration: string };
}

/**
 * The templates sub-page identity. Its body — the catalog cards and their
 * launch behavior — is this host's existing catalog surface.
 */
export interface InterfaceTemplatesPage {
  title: string;
  description: string;
}

export interface InterfaceManifest {
  version: typeof INTERFACE_VERSION;
  routes: InterfaceRoutes;
  navigation: {
    sidebar: { label: string };
    commandMenu: { title: string; description: string; keywords: string };
    /** The ordered sub-page navigation of the interface. */
    subPages?: InterfaceSubPageNavItem[];
  };
  pages: {
    list: {
      title: string;
      subtitle: string;
      overview?: InterfaceOverview;
      /** The filter dropdowns of the list page, in render order. */
      filters?: InterfaceDashboardFilter[];
      sort?: InterfaceDashboardSort;
      insights?: InterfaceListInsights;
    };
    detail: { backLabel: string };
    edit: { title: string };
    templates?: InterfaceTemplatesPage;
  };
  docsUrl: string;
  /**
   * The input surface of an existing Automation, keyed by the runtime-model
   * property the host sends. Rendering it as an edit dialog is this host's
   * choice, not stated here.
   */
  attributes: Partial<Record<AutomationAttributeName, InterfaceAttribute>>;
  importExport: InterfaceImportExport;
  endpoints: InterfaceEndpoints;
  featuredAutomationIds: string[];
  responderIntegrationIds: string[];
}
