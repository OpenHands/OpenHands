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

export interface Automation {
  id: string;
  name: string;
  trigger: AutomationTrigger;
  enabled: boolean;
  repository?: string;
  /** LLM/model profile name used for automation runs. */
  model?: string | null;
  /**
   * Maximum run time in seconds. `null`/omitted uses the server default
   * (600s, 10 min); the deployment reports the maximum it accepts.
   */
  timeout?: number | null;

  created_at: string;
  updated_at: string;
  prompt: string | null;
  branch?: string;
  plugins?: string[];
  notification?: string;
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
  error_detail: string | null;
  started_at: string;
  completed_at: string | null;
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
}
