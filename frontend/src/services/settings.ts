import { Settings } from "#/types/settings";

export const LATEST_SETTINGS_VERSION = 5;

export const DEFAULT_SETTINGS: Settings = {
  LLM_MODEL: "openhands/claude-sonnet-4-20250514",
  LLM_BASE_URL: "",
  AGENT: "CodeActAgent",
  LANGUAGE: "en",
  LLM_API_KEY_SET: false,
  SEARCH_API_KEY_SET: false,
  CONFIRMATION_MODE: false,
  SECURITY_ANALYZER: "llm",
  REMOTE_RUNTIME_RESOURCE_FACTOR: 1,
  PROVIDER_TOKENS_SET: {},
  ENABLE_DEFAULT_CONDENSER: true,
  CONDENSER_MAX_SIZE: 120,
  ENABLE_SOUND_NOTIFICATIONS: false,
  USER_CONSENTS_TO_ANALYTICS: false,
  ENABLE_PROACTIVE_CONVERSATION_STARTERS: false,
  ENABLE_SOLVABILITY_ANALYSIS: false,
  SEARCH_API_KEY: "",
  IS_NEW_USER: true,
  MAX_BUDGET_PER_TASK: null,
  EMAIL: "",
  EMAIL_VERIFIED: true, // Default to true to avoid restricting access unnecessarily
  MCP_CONFIG: {
    sse_servers: [],
    stdio_servers: [],
    shttp_servers: [],
  },
  GIT_USER_NAME: "openhands",
  GIT_USER_EMAIL: "openhands@all-hands.dev",
  V1_ENABLED: false,
};

/**
 * Get the default settings
 */
export const getDefaultSettings = (): Settings => DEFAULT_SETTINGS;
