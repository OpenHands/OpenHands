import type {
  ROUTING_GOALS,
  ROUTING_MODES,
  ROUTING_PRESETS,
  ROUTING_RETENTION_VALUES,
  ROUTING_TARGET_AUTO,
  ROUTING_WATERMARK_VALUES,
} from "./routing-constants";

export type RoutingGoal = (typeof ROUTING_GOALS)[number];
export type RoutingMode = (typeof ROUTING_MODES)[number];
export type RoutingPreset = (typeof ROUTING_PRESETS)[number];
export type RoutingRetention = (typeof ROUTING_RETENTION_VALUES)[number];
export type RoutingWatermark = (typeof ROUTING_WATERMARK_VALUES)[number];

export interface RoutingGuardrails {
  forbid_training_retention: boolean;
  forbid_watermarking: boolean;
  max_cost_usd_per_task: number | null;
  max_latency_s: number | null;
}

export interface RoutingLabel {
  id: string;
  name: string;
  description: string;
}

export type RoutingTarget =
  | typeof ROUTING_TARGET_AUTO
  | { provider_key: string; model: string };

export interface RoutingRoute {
  id: string;
  work_type: string | null;
  sensitivity: string | null;
  goal: RoutingGoal;
  target: RoutingTarget;
  guardrails: RoutingGuardrails;
}

export interface RoutingRouterModelConfig {
  preset: RoutingPreset;
  disabled: boolean;
  provider_key: string | null;
  model: string | null;
  goal: RoutingGoal;
  guardrails: RoutingGuardrails;
}

export interface RoutingConfig {
  goal: RoutingGoal;
  guardrails: RoutingGuardrails;
  mode: RoutingMode;
  metadata_max_age_days: number;
  blend_alpha: number;
  cost_mode: "floor" | "cap";
  quality_floor: number;
  struggle_threshold: number;
  router_model: RoutingRouterModelConfig;
  routes: RoutingRoute[];
  work_types: RoutingLabel[];
  sensitivities: RoutingLabel[];
  imported_project_path: string | null;
}

export interface RoutingTaxonomy {
  work_types: RoutingLabel[];
  sensitivities: RoutingLabel[];
  classifier_prompt: string;
}

export interface RoutingProvenance {
  source: string;
  fetched_at: string;
  version: string | null;
}

export interface RoutingBenchmarkEntry {
  score: number;
  provenance: RoutingProvenance | null;
}

export interface RoutingRegistryModel {
  id: string;
  provider_key: string;
  benchmarks: Partial<
    Record<"coding" | "reasoning" | "ux" | "copy", RoutingBenchmarkEntry>
  >;
  cost_per_1k: number;
  latency_s_p90: number;
  local: boolean;
  runtime: string | null;
  source_url: string | null;
  verified: boolean;
  notes: string | null;
  retention: RoutingRetention;
  watermark: RoutingWatermark;
  reachable?: boolean;
}

export interface RoutingRegistrySnapshot {
  version: string;
  last_updated: string;
  privacy_last_updated: string;
  stale: boolean;
  privacy_stale: boolean;
  models: RoutingRegistryModel[];
  sources: Record<string, RoutingSourceStatus>;
  curated_fields?: string;
  connected_providers?: string[];
}

export interface RoutingSourceStatus {
  id?: string;
  last_success: string | null;
  last_error: string | null;
  stale: boolean;
  version?: string | null;
  rows?: number;
}

export interface RoutingSourcesResponse {
  sources: RoutingSourceStatus[];
}

export interface RoutingTradeoffs {
  cost_per_1k: number | null;
  latency_s_p90: number | null;
  retention: RoutingRetention | null;
  watermark: RoutingWatermark | null;
  classification_accuracy_proxy: number;
  offline_capable: boolean;
  verified: boolean;
}

export interface RoutingPresetResolved {
  preset: RoutingPreset;
  provider_key: string;
  model: string;
  goal: RoutingGoal;
  guardrails: RoutingGuardrails;
  tradeoffs: RoutingTradeoffs;
  pool: Array<
    {
      id: string;
      provider_key: string;
    } & RoutingTradeoffs
  >;
}

export interface RoutingRouterModelResponse {
  config: RoutingRouterModelConfig;
  resolved: RoutingPresetResolved;
  runtimes?: RoutingLocalRuntimes;
}

export interface RoutingLocalRuntimeInfo {
  alive: boolean;
  models: string[];
  error: string | null;
}

export interface RoutingLocalRuntimes {
  runtimes: Record<string, RoutingLocalRuntimeInfo>;
  best_by_category?: Record<
    string,
    { id: string; provider_key: string; score: number }
  >;
}

export interface RoutingClassification {
  work_type: string;
  sensitivity: string;
  complexity: string;
  confidence: number;
  reason: string;
  classifier?: string;
  classifier_version?: string;
}

export interface RoutingDroppedCandidate {
  id: string;
  reason: string;
}

export interface RoutingRankedCandidate {
  id: string;
  provider_key: string;
  score: number;
  score_source: string;
  cost_per_1k?: number;
  latency_s_p90?: number;
  retention?: string;
  watermark?: string;
}

export interface RoutingDecision {
  provider_key: string | null;
  model: string | null;
  score: number | null;
  score_source?: string;
  rule_id: string | null;
  registry_version: string;
  target: RoutingTarget;
  usable: boolean;
  goal: RoutingGoal;
  locked?: boolean;
  warning?: string | null;
}

export interface RoutingTrace {
  task_text: string;
  classification: RoutingClassification;
  classifier_version: string;
  route_id: string;
  filters: RoutingDroppedCandidate[];
  ranked: RoutingRankedCandidate[];
  chosen: RoutingDecision;
  reason: string;
  registry_last_updated?: string;
  privacy_last_updated?: string;
  privacy_stale?: boolean;
}

export interface RoutingResolveResult {
  decision: RoutingDecision;
  trace: RoutingTrace;
  audit_id?: string;
}

export interface RoutingAuditItem {
  id: string;
  created_at: string;
  kind: "resolve" | "switch";
  card_id: string | null;
  run_id: string | null;
  payload: {
    decision?: RoutingDecision;
    trace?: RoutingTrace;
    from?: RoutingDecision;
    to?: RoutingDecision;
    reason?: string;
  };
}

export interface RoutingAuditPage {
  items: RoutingAuditItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface RoutingResolveRequest {
  task_text: string;
  work_type?: string;
  sensitivity?: string;
  connected_providers?: string[];
  card_id?: string;
  run_id?: string;
  local_runtimes?: Record<string, RoutingLocalRuntimeInfo>;
}

export interface RoutingIngestResult {
  sources: Record<
    string,
    { ok: boolean; error?: string; rows?: number; unmapped?: number }
  >;
  unmapped: Array<{ name: string; source: string }>;
}
