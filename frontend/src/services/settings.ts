import { Settings } from "#/types/settings";

export const LATEST_SETTINGS_VERSION = 5;

export const DEFAULT_SETTINGS: Settings = {
  llm_model: "openhands/claude-opus-4-5-20251101",
  llm_base_url: "",
  agent: "CodeActAgent",
  language: "en",
  llm_api_key: null,
  llm_api_key_set: false,
  search_api_key_set: false,
  confirmation_mode: false,
  security_analyzer: "llm",
  max_iterations: null,
  remote_runtime_resource_factor: 1,
  provider_tokens_set: {},
  enable_default_condenser: true,
  condenser_max_size: 240,
  enable_sound_notifications: false,
  user_consents_to_analytics: false,
  enable_proactive_conversation_starters: false,
  enable_solvability_analysis: false,
  search_api_key: "",
  is_new_user: true,
  disabled_skills: [],
  mcp_config: {
    sse_servers: [],
    stdio_servers: [],
    shttp_servers: [],
  },
  max_budget_per_task: null,
  email: "",
  email_verified: true,
  git_user_name: "openhands",
  git_user_email: "openhands@all-hands.dev",
  v1_enabled: true,
  sandbox_grouping_strategy: "NO_GROUPING",
  agent_settings_schema: null,
  agent_settings: {
    schema_version: 1,
    agent: "CodeActAgent",
    llm: {
      model: "openhands/claude-opus-4-5-20251101",
    },
    condenser: {
      enabled: true,
      max_size: 240,
    },
    mcp_config: {
      sse_servers: [],
      stdio_servers: [],
      shttp_servers: [],
    },
  },
  conversation_settings_schema: null,
  conversation_settings: {
    schema_version: 1,
    confirmation_mode: false,
    security_analyzer: "llm",
  },
};

/**
 * Get the default settings
 */
export const getDefaultSettings = (): Settings => DEFAULT_SETTINGS;
