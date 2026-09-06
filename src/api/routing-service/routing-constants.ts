export const ROUTING_PATH = "/routing";
export const ROUTING_API_PATH = "/api/routing";
export const SESSION_API_KEY_HEADER = "X-Session-API-Key";

export const ROUTING_CONFIG_PATH = `${ROUTING_API_PATH}/config`;
export const ROUTING_IMPORT_PATH = `${ROUTING_API_PATH}/import-project-config`;
export const ROUTING_TAXONOMY_PATH = `${ROUTING_API_PATH}/taxonomy`;
export const ROUTING_REGISTRY_PATH = `${ROUTING_API_PATH}/registry`;
export const ROUTING_INGEST_PATH = `${ROUTING_API_PATH}/benchmarks/ingest`;
export const ROUTING_SOURCES_PATH = `${ROUTING_API_PATH}/benchmarks/sources`;
export const ROUTING_PRIVACY_REFRESH_PATH = `${ROUTING_API_PATH}/registry/privacy-refresh`;
export const ROUTING_ROUTER_MODEL_PATH = `${ROUTING_API_PATH}/router-model`;
export const ROUTING_LOCAL_RUNTIMES_PATH = `${ROUTING_API_PATH}/local-runtimes`;
export const ROUTING_RESOLVE_PATH = `${ROUTING_API_PATH}/resolve`;
export const ROUTING_AUDIT_PATH = `${ROUTING_API_PATH}/audit`;

export const ROUTING_GOALS = ["quality", "cost", "speed", "privacy"] as const;

export const ROUTING_MODES = ["warn", "strict"] as const;

export const ROUTING_PRESETS = [
  "local",
  "best-intelligence",
  "cheapest",
  "most-secure",
  "custom",
] as const;

export const ROUTING_TARGET_AUTO = "auto";

export const ROUTING_RETENTION_VALUES = ["none", "opt-out", "trains"] as const;
export const ROUTING_WATERMARK_VALUES = [
  "none",
  "configurable",
  "always",
] as const;

export const ROUTING_BENCHMARK_CATEGORIES = [
  "coding",
  "reasoning",
  "ux",
  "copy",
] as const;
