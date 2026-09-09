export const STANDARDS_PATH = "/standards";
export const STANDARDS_API_PATH = "/api/standards";
export const SESSION_API_KEY_HEADER = "X-Session-API-Key";

export const STANDARDS_PLUGINS_PATH = `${STANDARDS_API_PATH}/plugins`;
export const STANDARDS_CONFIG_PATH = `${STANDARDS_API_PATH}/config`;
export const STANDARDS_RUN_PATH = `${STANDARDS_API_PATH}/run`;
export const STANDARDS_AUDIT_PATH = `${STANDARDS_API_PATH}/audit`;

export const STANDARDS_ACTION_WARN = "warn";
export const STANDARDS_ACTION_BLOCK = "block";
export const STANDARDS_ACTIONS = [
  STANDARDS_ACTION_WARN,
  STANDARDS_ACTION_BLOCK,
] as const;

export const STANDARDS_SEVERITY_INFO = "info";
export const STANDARDS_SEVERITY_WARNING = "warning";
export const STANDARDS_SEVERITY_ERROR = "error";
export const STANDARDS_SEVERITIES = [
  STANDARDS_SEVERITY_INFO,
  STANDARDS_SEVERITY_WARNING,
  STANDARDS_SEVERITY_ERROR,
] as const;

export const STANDARDS_SOURCE_BUILTIN = "builtin";
export const STANDARDS_SOURCE_USER = "user";
export const STANDARDS_SOURCE_PROJECT = "project";
export const STANDARDS_SOURCES = [
  STANDARDS_SOURCE_BUILTIN,
  STANDARDS_SOURCE_USER,
  STANDARDS_SOURCE_PROJECT,
] as const;

export const DEFAULT_STANDARDS_AUDIT_LIMIT = 50;
