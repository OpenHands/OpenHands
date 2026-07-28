/**
 * Host-owned shape of an extension manifest.
 *
 * A manifest is declarative data authored in another repository that tells this
 * host what to ask a user for and which pre-approved capability to invoke with
 * the answers. The host never originates any of that knowledge itself, so these
 * types deliberately carry no vocabulary from any particular feature.
 *
 * The names are structural, so a manifest published by an extension package
 * assigns to `ExtensionManifest` without an adapter.
 */

export const MANIFEST_VERSION = "1.0";

export type ManifestSetupMode = "direct" | "assisted";

export type ManifestFieldType =
  | "text"
  | "textarea"
  | "select"
  | "cron"
  | "repo-picker";

export type ManifestGitProvider = "github" | "gitlab" | "bitbucket";

export type ManifestTriggerKind = "cron" | "event";

/**
 * Actions this host is willing to perform on a manifest's behalf. The manifest
 * chooses among pre-approved capabilities; it cannot describe an arbitrary
 * request. Extending this list is a deliberate host decision.
 */
export const MANIFEST_ACTIONS = [
  "automation.create",
  "conversation.start",
] as const;

export type ManifestAction = (typeof MANIFEST_ACTIONS)[number];

/** Placeholder namespaces a manifest may reference inside `{{...}}`. */
export const MANIFEST_PLACEHOLDER_NAMESPACES = [
  "form",
  "manifest",
  "response",
  "capabilities",
  "submit",
] as const;

export type ManifestAnalyticsEvent =
  | "route.entered"
  | "capabilities.resolved"
  | "validation.succeeded"
  | "submit.succeeded"
  | "submit.failed";

export interface ManifestRoute {
  path: string;
  page: "setup";
}

export interface ManifestFieldOption {
  value: string;
  label: string;
}

export interface ManifestFieldConstraints {
  minLength?: number;
  maxLength?: number;
  /**
   * A host-implemented check named from a closed set. Manifests supply no
   * regex of their own, so they cannot hand the host a pathological pattern.
   */
  format?: "safeExpressionLiteral";
}

export interface ManifestFormField {
  name: string;
  type: ManifestFieldType;
  label: string;
  help: string;
  placeholder?: string;
  default?: string;
  required: boolean;
  provider?: ManifestGitProvider;
  options?: ManifestFieldOption[];
  constraints?: ManifestFieldConstraints;
}

export interface ManifestForm {
  note?: string;
  fields: ManifestFormField[];
}

export interface ManifestCapabilityRequirements {
  triggerKinds?: ManifestTriggerKind[];
  eventSources?: string[];
  eventTypes?: string[];
  features?: string[];
  ready?: true;
}

export interface ManifestCapabilityBinding {
  field: string;
  constraint: "options" | "minIntervalSeconds";
  /** Dotted path into the capabilities response. */
  from: string;
}

export interface ManifestCapabilities {
  discovery: { method: "GET"; path: string };
  requires: ManifestCapabilityRequirements;
  bindings?: ManifestCapabilityBinding[];
  onUnsupported: { behavior: "block"; message: string };
}

export interface ManifestIntegrationRequirement {
  id: string;
  reason: string;
  enforcement: "block" | "warn";
}

/** Credential names only. A manifest never carries a credential value. */
export interface ManifestSecretRequirement {
  key: string;
  label: string;
  help: string;
  required: boolean;
}

export interface ManifestPrerequisites {
  integrations: ManifestIntegrationRequirement[];
  secrets: ManifestSecretRequirement[];
  onUnmet: { behavior: "block"; message: string };
  onWarn?: { behavior: "continue"; message: string };
}

export type ManifestPayloadValue =
  | string
  | number
  | boolean
  | null
  | ManifestPayloadValue[]
  | { [key: string]: ManifestPayloadValue };

export interface ManifestRequestBody {
  [key: string]: ManifestPayloadValue;
}

export interface ManifestPreflight {
  method: "POST";
  path: string;
  runOn: ("fieldBlur" | "beforeSubmit")[];
  debounceMs?: number;
  body: ManifestRequestBody;
}

export interface ManifestValidation {
  /** Omitted when local validation is the only check available before submit. */
  preflight?: ManifestPreflight;
  onInvalid: {
    behavior: "blockSubmit";
    errorTarget: "field" | "form";
    /** Payload path to the form field, or fields, that produced it. */
    errorMap?: Record<string, string | string[]>;
  };
}

export interface ManifestReviewRow {
  label: string;
  value: string;
}

export interface ManifestReview {
  title: string;
  note?: string;
  emptyValueText?: string;
  summary: ManifestReviewRow[];
  confirmLabel: string;
}

export interface ManifestSubmitOnSuccess {
  behavior: "navigate";
  to: string;
}

export interface ManifestSubmitOnError {
  behavior: "stayOnForm";
  errorTarget: "field" | "form";
  reuseErrorMap?: boolean;
  message?: string;
}

export interface ManifestDirectSubmit {
  action: "automation.create";
  endpoint: { method: "POST"; path: string };
  payload: ManifestRequestBody;
  onSuccess: ManifestSubmitOnSuccess;
  onError: ManifestSubmitOnError;
}

export interface ManifestAssistedSubmit {
  action: "conversation.start";
  message: string;
  onSuccess: ManifestSubmitOnSuccess;
  onError: ManifestSubmitOnError;
}

export type ManifestSubmit = ManifestDirectSubmit | ManifestAssistedSubmit;

export interface ManifestAnalyticsStage {
  id: string;
  on: ManifestAnalyticsEvent;
  properties: Record<string, string | number | boolean>;
}

export interface ManifestAnalytics {
  consent: "required";
  stages: ManifestAnalyticsStage[];
}

export interface ExtensionManifest {
  manifestVersion: typeof MANIFEST_VERSION;
  id: string;
  name: string;
  category: string;
  description: string;
  setupMode: ManifestSetupMode;
  routes: ManifestRoute[];
  capabilities?: ManifestCapabilities;
  requires?: ManifestPrerequisites;
  form: ManifestForm;
  validation?: ManifestValidation;
  review: ManifestReview;
  submit: ManifestSubmit;
  analytics: ManifestAnalytics;
}

/** Field-addressable error returned by preflight validation. */
export interface ManifestPreflightError {
  field: string;
  code?: string;
  message: string;
}

export interface ManifestPreflightResponse {
  valid: boolean;
  errors: ManifestPreflightError[];
}

/** Form values are collected as strings; the payload mapping shapes them. */
export type ManifestFormValues = Record<string, string>;
