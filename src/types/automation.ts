export interface AutomationTrigger {
  /**
   * Trigger kind. Known values are the schedule aliases "cron" / "schedule"
   * (time-based) and "event" (webhook/event-driven). Kept as `string` rather
   * than a closed union on purpose: the backend emits more than one
   * scheduled-trigger alias and may introduce new kinds, so UI code branches
   * on `type === "event"` and treats every other value as a schedule.
   */
  type: string;
  /** Cron expression (schedule triggers only). */
  schedule?: string;
  /** Human-readable schedule description (schedule triggers only). */
  schedule_human?: string;
  /** IANA timezone name (schedule triggers only). */
  timezone?: string;
  /** Event source, e.g. "github" (event triggers only). */
  source?: string;
  /** Event key pattern(s) to match, e.g. "pull_request.opened" or ["push", "release.*"]. */
  on?: string | string[];
  /** JMESPath filter expression evaluated against the raw webhook payload. */
  filter?: string;
}

/** A single repository to clone, as sent to and echoed back by the backend. */
export interface AutomationRepoSource {
  url: string;
  /** Branch, tag, or commit SHA to checkout. Omitted means the default branch. */
  ref?: string;
  provider?: "github" | "gitlab" | "bitbucket";
}

/**
 * Opaque provenance the backend stores verbatim under `preset_metadata`.
 * Only the keys the UI actually reads are typed here; the backend may store
 * more (e.g. `template`) that this UI does not surface.
 */
export interface AutomationPresetMetadata {
  /** Repos cloned for this automation. Absent/empty means none configured. */
  repos?: AutomationRepoSource[];
  /** Plugin identifiers baked into the tarball, for plugin-based automations. */
  plugins?: string[];
}

export interface Automation {
  id: string;
  name: string;
  trigger: AutomationTrigger;
  enabled: boolean;
  /**
   * Single-repo display fields kept for backward compatibility with
   * automations created before multi-repo support. Prefer
   * `preset_metadata.repos` when present — it carries the full repo list.
   */
  repository?: string;
  branch?: string;
  /** LLM/model profile name used for automation runs. */
  model?: string | null;
  /**
   * Maximum run time in seconds. `null`/omitted uses the server default
   * (600s, 10 min); the deployment reports the maximum it accepts.
   */
  timeout?: number | null;
  /**
   * If true, the sandbox is left for runtime TTL cleanup after the run
   * finishes instead of being torn down immediately. `null`/omitted means
   * explicit cleanup (the server default).
   */
  keep_alive?: boolean | null;
  /** Repo/template/plugin provenance recorded at creation time. */
  preset_metadata?: AutomationPresetMetadata | null;

  created_at: string;
  updated_at: string;
  prompt: string | null;
  plugins?: string[];
  timezone?: string;
  last_triggered_at?: string | null;
}

export type AutomationSpec = Omit<
  Automation,
  "id" | "created_at" | "updated_at" | "last_triggered_at"
>;

/** The envelope constants come from the interface manifest's import/export spec. */
export interface AutomationExportFile {
  version: number;
  kind: string;
  spec: AutomationSpec;
}

export interface AutomationsResponse {
  automations: Automation[];
  total: number;
}

/** Mirrors `RunStatus` in the automation service's OpenAPI schema. */
export enum AutomationRunStatus {
  PENDING = "PENDING",
  RUNNING = "RUNNING",
  COMPLETED = "COMPLETED",
  FAILED = "FAILED",
  CANCELLED = "CANCELLED",
  SKIPPED = "SKIPPED",
}

export interface AutomationRun {
  id: string;
  /**
   * ID of the automation this run belongs to. Optional: list-by-automation
   * responses are already scoped to one automation and some older backend
   * versions omit it there.
   */
  automation_id?: string;
  status: AutomationRunStatus;
  conversation_id: string | null;
  /**
   * ID of the bash command that ran the automation inside the agent-server
   * sandbox. Used to fetch run logs from
   * `/api/bash/bash_events/{bash_command_id}` and the matching
   * `BashOutput` events. Null when the run failed before a command was
   * dispatched (e.g. sandbox provisioning errors).
   */
  bash_command_id: string | null;
  /** ID of the sandbox the run executed in. Null before one is provisioned. */
  sandbox_id?: string | null;
  error_detail: string | null;
  /**
   * Accumulated LLM cost of the run in USD, reported by the SDK in the
   * completion callback. `null` means unknown — the run predates cost
   * tracking, or ended without a callback (cancelled, watchdog timeout).
   * Absent entirely when the automation service is older than the release
   * that added the field, hence optional.
   */
  cost?: number | null;
  /** When the run was created (queued). Optional for older backends. */
  created_at?: string;
  started_at: string;
  completed_at: string | null;
  /** Deadline the run must complete by. Null when the automation has no timeout. */
  timeout_at?: string | null;
}

export interface AutomationRunsResponse {
  runs: AutomationRun[];
  total: number;
}

export type ActivityLogExportFormat = "json" | "csv";

/** Client-built Activity Log export row (from list runs + automation detail). */
export interface AutomationRunExportRow {
  run_id: string;
  automation_id: string;
  automation_name: string;
  trigger: AutomationTrigger | Record<string, unknown>;
  start_time: string | null;
  end_time: string | null;
  duration_seconds: number | null;
  status: AutomationRunStatus;
  conversation_id: string | null;
  conversation_url: string | null;
  error: string | null;
  /**
   * Accumulated LLM cost in USD, or null when unknown. Unlike
   * `AutomationRun["cost"]` this is always present: the row normalizes a
   * missing field to null so every exported record has the same shape.
   */
  cost: number | null;
}
