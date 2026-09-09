export const GRAPH_PATH = "/settings/agent-context";
export const GRAPH_API_PATH = "/api/graph";
export const SESSION_API_KEY_HEADER = "X-Session-API-Key";

export const GRAPH_INDEX_PATH = `${GRAPH_API_PATH}/index`;
export const GRAPH_STATUS_PATH = `${GRAPH_API_PATH}/index/status`;
export const GRAPH_RETRIGGER_PATH = `${GRAPH_API_PATH}/index/retrigger`;
export const GRAPH_QUERY_PATH = `${GRAPH_API_PATH}/query`;
export const GRAPH_DEFINITIONS_PATH = `${GRAPH_API_PATH}/definitions`;
export const GRAPH_CONFIG_PATH = `${GRAPH_API_PATH}/config`;
export const GRAPH_IMPORT_PATH = `${GRAPH_API_PATH}/import-project-config`;

export const GRAPH_QUERY_CALLERS = "callers";
export const GRAPH_QUERY_DEPS = "deps";
export const GRAPH_QUERY_USAGES = "usages";

export const GRAPH_QUERIES = [
  GRAPH_QUERY_CALLERS,
  GRAPH_QUERY_DEPS,
  GRAPH_QUERY_USAGES,
] as const;

export const GRAPH_SOURCE = "graph";

export const GRAPH_LANGUAGES = [
  "python",
  "javascript",
  "typescript",
  "markdown",
] as const;
