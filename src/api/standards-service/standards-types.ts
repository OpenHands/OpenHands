import type {
  STANDARDS_ACTIONS,
  STANDARDS_SEVERITIES,
  STANDARDS_SOURCES,
} from "./standards-constants";

export type StandardsAction = (typeof STANDARDS_ACTIONS)[number];
export type StandardsSeverity = (typeof STANDARDS_SEVERITIES)[number];
export type StandardsSource = (typeof STANDARDS_SOURCES)[number];

export interface StandardsPluginInfo {
  name: string;
  display_name: string;
  description: string;
  version: string;
  source: StandardsSource;
  enabled: boolean;
  action: StandardsAction;
  severity_default: StandardsSeverity;
}

export interface StandardsLoadError {
  path: string;
  message: string;
  source: string;
}

export interface StandardsPluginsResponse {
  plugins: StandardsPluginInfo[];
  load_errors: StandardsLoadError[];
}

export interface StandardsEnforcement {
  prompt: boolean;
  automated: boolean;
  gates: boolean;
}

export interface StandardsPluginConfig {
  name: string;
  enabled: boolean;
  action: StandardsAction;
}

export interface StandardsConfig {
  enabled: boolean;
  enforcement: StandardsEnforcement;
  plugins: StandardsPluginConfig[];
  project_yaml_source: boolean;
}

export interface StandardsViolation {
  plugin_name: string;
  rule_id: string;
  severity: StandardsSeverity;
  file: string;
  line: number | null;
  message: string;
  remediation: string;
  action: StandardsAction;
  fixable?: boolean;
}

export interface StandardsRunSummary {
  files_scanned: number;
  violation_count: number;
  info: number;
  warning: number;
  error: number;
}

export interface StandardsRunResult {
  run_id: string;
  summary: StandardsRunSummary;
  violations: StandardsViolation[];
  duration_ms: number;
  status: string;
  started_at?: string;
  worktree?: string;
}

export interface StandardsAuditItem {
  id: string;
  run_id: string;
  created_at: string;
  plugin_name: string;
  rule_id: string;
  severity: StandardsSeverity;
  file: string;
  line: number | null;
  message: string;
  remediation: string;
  action: StandardsAction;
}

export interface StandardsAuditPage {
  items: StandardsAuditItem[];
  limit: number;
  next_before_id: string | null;
}

export interface StandardsAuditParams {
  run_id?: string;
  plugin?: string;
  severity?: string;
  file?: string;
  limit?: number;
  before_id?: string;
}

export interface StandardsRunRequest {
  root: string;
  enabled_names?: string[];
  files?: string[];
}
