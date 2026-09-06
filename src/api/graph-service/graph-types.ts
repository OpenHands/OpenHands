import type { GRAPH_LANGUAGES, GRAPH_QUERIES } from "./graph-constants";

export type GraphQueryKind = (typeof GRAPH_QUERIES)[number];
export type GraphLanguage = (typeof GRAPH_LANGUAGES)[number];

export interface GraphIndexStatus {
  running: boolean;
  files_indexed: number;
  symbols: number;
  edges: number;
  languages_used: string[];
  coverage: number;
  last_full_index_at: string | null;
  last_error: string | null;
  root?: string | null;
  files_parsed?: number;
}

export interface GraphConfig {
  enabled: boolean;
  languages: string[];
  graph_budget_lines: number;
  max_context_files: number;
  strict: boolean;
  stale_after_minutes: number;
  imported_project_path: string | null;
}

export interface GraphNode {
  id: string;
  kind: string;
  name: string;
  path?: string | null;
  start_line?: number | null;
  end_line?: number | null;
  external?: boolean;
  qualified_name?: string;
}

export interface GraphQueryResult {
  query: string;
  symbol: string | null;
  file?: string | null;
  result: GraphNode[];
  source: "graph";
  status: "ok" | "stale" | "empty";
  last_full_index_at: string | null;
  coverage: number;
}

export interface GraphIndexRequest {
  root?: string;
  full?: boolean;
}

export interface GraphQueryParams {
  q: GraphQueryKind;
  symbol?: string;
  file?: string;
  root?: string;
  budget?: number;
}
