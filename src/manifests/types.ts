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

export interface InterfaceManifest {
  version: typeof INTERFACE_VERSION;
  routes: InterfaceRoutes;
  navigation: {
    sidebar: { label: string };
    commandMenu: { title: string; description: string; keywords: string };
  };
  pages: {
    list: { title: string; subtitle: string };
    detail: { backLabel: string };
    edit: { title: string };
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
