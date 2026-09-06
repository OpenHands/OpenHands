import {
  LOOP_RUN_STATUSES,
  LOOP_SCHEDULE_TYPES,
  LOOP_TRIGGER_TYPES,
  TRIGGER_EVENT_STATUSES,
} from "./loop-constants";

export type LoopTriggerType = (typeof LOOP_TRIGGER_TYPES)[number];
export type LoopScheduleType = (typeof LOOP_SCHEDULE_TYPES)[number];
export type LoopRunStatus = (typeof LOOP_RUN_STATUSES)[number];
export type TriggerEventStatus = (typeof TRIGGER_EVENT_STATUSES)[number];

export interface LoopStage {
  name: string;
  cmd: string | null;
  iterative: boolean;
}

export interface LoopDefinition {
  id: string;
  name: string;
  project_id: string;
  stages: LoopStage[];
  max_iterations: number;
  max_cost_usd: number;
  on_failure: string;
  config: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface LoopStageRun {
  id: string;
  loop_run_id: string;
  stage_name: string;
  status: string;
  attempt: number;
  last_output: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface LoopRun {
  id: string;
  definition_id: string;
  project_id: string;
  session_id: string | null;
  worktree_dir: string | null;
  status: LoopRunStatus;
  current_stage: string | null;
  iteration: number;
  total_cost_usd: number;
  created_at: string;
  updated_at: string;
  stages: LoopStageRun[];
}

export interface LoopTrigger {
  id: string;
  project_id: string;
  loop_definition_id: string;
  trigger_type: LoopTriggerType;
  schedule_type: LoopScheduleType | null;
  cron_expr: string | null;
  interval_seconds: number | null;
  payload: Record<string, unknown>;
  enabled: boolean;
  last_fired_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface TriggerEvent {
  id: string;
  trigger_id: string;
  loop_run_id: string | null;
  fired_at: string;
  status: TriggerEventStatus;
  reason: string | null;
}

export interface CreateLoopTriggerPayload {
  project_id: string;
  loop_definition_id: string;
  trigger_type: LoopTriggerType;
  schedule_type?: LoopScheduleType | null;
  cron_expr?: string | null;
  interval_seconds?: number | null;
  payload?: Record<string, unknown>;
  enabled?: boolean;
}

export interface UpdateLoopTriggerPayload {
  enabled?: boolean;
  schedule_type?: LoopScheduleType | null;
  cron_expr?: string | null;
  interval_seconds?: number | null;
  payload?: Record<string, unknown>;
}

export interface FireTriggerResult {
  event: TriggerEvent;
  run: LoopRun;
}
