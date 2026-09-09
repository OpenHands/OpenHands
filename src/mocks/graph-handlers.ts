import { http, HttpResponse } from "msw";
import {
  GRAPH_CONFIG_PATH,
  GRAPH_DEFINITIONS_PATH,
  GRAPH_IMPORT_PATH,
  GRAPH_INDEX_PATH,
  GRAPH_LANGUAGES,
  GRAPH_QUERY_PATH,
  GRAPH_RETRIGGER_PATH,
  GRAPH_SOURCE,
  GRAPH_STATUS_PATH,
} from "#/api/graph-service/graph-constants";
import type {
  GraphConfig,
  GraphIndexStatus,
  GraphQueryResult,
} from "#/api/graph-service/graph-types";

const DEFAULT_STATUS: GraphIndexStatus = {
  running: false,
  files_indexed: 4,
  symbols: 6,
  edges: 8,
  languages_used: [...GRAPH_LANGUAGES],
  coverage: 1,
  last_full_index_at: "2026-01-01T00:00:00+00:00",
  last_error: null,
  root: "/workspace/project",
};

const DEFAULT_CONFIG: GraphConfig = {
  enabled: true,
  languages: [...GRAPH_LANGUAGES],
  graph_budget_lines: 400,
  max_context_files: 12,
  strict: false,
  stale_after_minutes: 60,
  imported_project_path: null,
};

let status: GraphIndexStatus = { ...DEFAULT_STATUS };
let config: GraphConfig = { ...DEFAULT_CONFIG };

export function resetGraphMockData() {
  status = { ...DEFAULT_STATUS };
  config = { ...DEFAULT_CONFIG };
}

function queryResult(query: string, symbol: string | null): GraphQueryResult {
  return {
    query,
    symbol,
    result: [
      {
        id: "function:app.py:main:4",
        kind: "function",
        name: "main",
        path: "app.py",
        start_line: 4,
        end_line: 6,
        external: false,
      },
    ],
    source: GRAPH_SOURCE,
    status: "ok",
    last_full_index_at: status.last_full_index_at,
    coverage: status.coverage,
  };
}

export const GRAPH_HANDLERS = [
  http.post(`*${GRAPH_INDEX_PATH}`, async ({ request }) => {
    const body = (await request.json().catch(() => ({}))) as {
      full?: boolean;
      root?: string;
    };
    status = {
      ...status,
      running: false,
      files_indexed: body.full ? 5 : status.files_indexed,
      last_error: null,
      root: body.root ?? status.root,
    };
    return HttpResponse.json(status);
  }),
  http.get(`*${GRAPH_STATUS_PATH}`, () => HttpResponse.json(status)),
  http.delete(`*${GRAPH_INDEX_PATH}`, () => {
    status = {
      ...status,
      files_indexed: 0,
      symbols: 0,
      edges: 0,
      coverage: 0,
      last_full_index_at: null,
    };
    return HttpResponse.json(status);
  }),
  http.post(`*${GRAPH_RETRIGGER_PATH}`, () => {
    status = {
      ...status,
      files_indexed: 5,
      symbols: 8,
      edges: 10,
      coverage: 1,
      last_full_index_at: "2026-01-02T00:00:00+00:00",
    };
    return HttpResponse.json(status);
  }),
  http.get(`*${GRAPH_QUERY_PATH}`, ({ request }) => {
    const url = new URL(request.url);
    return HttpResponse.json(
      queryResult(
        url.searchParams.get("q") ?? "callers",
        url.searchParams.get("symbol"),
      ),
    );
  }),
  http.get(`*${GRAPH_DEFINITIONS_PATH}`, ({ request }) => {
    const url = new URL(request.url);
    return HttpResponse.json(
      queryResult("definitions", url.searchParams.get("symbol")),
    );
  }),
  http.get(`*${GRAPH_CONFIG_PATH}`, () => HttpResponse.json(config)),
  http.put(`*${GRAPH_CONFIG_PATH}`, async ({ request }) => {
    const body = (await request.json()) as Partial<GraphConfig>;
    config = { ...config, ...body };
    return HttpResponse.json(config);
  }),
  http.post(`*${GRAPH_IMPORT_PATH}`, () => HttpResponse.json(config)),
];
