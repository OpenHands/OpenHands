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
  /** assisted only. Setup context for the conversation that finishes setup. */
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
