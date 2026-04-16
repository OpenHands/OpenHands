export const ProviderOptions = {
  github: "github",
  gitlab: "gitlab",
  bitbucket: "bitbucket",
  bitbucket_data_center: "bitbucket_data_center",
  azure_devops: "azure_devops",
  forgejo: "forgejo",
  enterprise_sso: "enterprise_sso",
} as const;

export const SandboxGroupingStrategyOptions = {
  NO_GROUPING: "NO_GROUPING",
  GROUP_BY_NEWEST: "GROUP_BY_NEWEST",
  LEAST_RECENTLY_USED: "LEAST_RECENTLY_USED",
  FEWEST_CONVERSATIONS: "FEWEST_CONVERSATIONS",
  ADD_TO_ANY: "ADD_TO_ANY",
} as const;

export type SandboxGroupingStrategy =
  keyof typeof SandboxGroupingStrategyOptions;

export type Provider = keyof typeof ProviderOptions;

export type ProviderToken = {
  token: string;
  host: string | null;
};

export type MCPSSEServer = {
  url: string;
  api_key?: string;
};

export type MCPStdioServer = {
  name: string;
  command: string;
  args?: string[];
  env?: Record<string, string>;
};

export type MCPSHTTPServer = {
  url: string;
  api_key?: string;
  timeout?: number;
};

export type MCPConfig = {
  sse_servers: (string | MCPSSEServer)[];
  stdio_servers: MCPStdioServer[];
  shttp_servers: (string | MCPSHTTPServer)[];
};

export type SettingsChoiceValue = boolean | number | string;

export type SettingsChoice = {
  label: string;
  value: SettingsChoiceValue;
};

export type SettingsValue =
  | boolean
  | number
  | string
  | null
  | SettingsValue[]
  | { [key: string]: SettingsValue };

export type SettingsValueType =
  | "string"
  | "integer"
  | "number"
  | "boolean"
  | "array"
  | "object";

export type SettingProminence = "critical" | "major" | "minor";

export type SettingsFieldSchema = {
  key: string;
  label: string;
  description?: string | null;
  section: string;
  section_label: string;
  value_type: SettingsValueType;
  default?: SettingsValue;
  choices: SettingsChoice[];
  depends_on: string[];
  prominence: SettingProminence;
  secret: boolean;
  required: boolean;
};

export type SettingsSectionSchema = {
  key: string;
  label: string;
  fields: SettingsFieldSchema[];
};

export type SettingsSchema = {
  model_name: string;
  sections: SettingsSectionSchema[];
};

export type SkillInfo = {
  name: string;
  type: string;
  source: string;
  triggers?: string[];
};

export type SettingsScope = "personal" | "org";

export type Settings = {
  llm_model: string;
  llm_base_url: string;
  agent: string;
  language: string;
  llm_api_key: string | null;
  llm_api_key_set: boolean;
  search_api_key_set: boolean;
  confirmation_mode: boolean;
  security_analyzer: string | null;
  max_iterations: number | null;
  remote_runtime_resource_factor: number | null;
  provider_tokens_set: Partial<Record<Provider, string | null>>;
  enable_default_condenser: boolean;
  condenser_max_size: number | null;
  enable_sound_notifications: boolean;
  enable_proactive_conversation_starters: boolean;
  enable_solvability_analysis: boolean;
  user_consents_to_analytics: boolean | null;
  search_api_key?: string;
  is_new_user?: boolean;
  mcp_config?: MCPConfig;
  disabled_skills?: string[];
  max_budget_per_task: number | null;
  email?: string;
  email_verified?: boolean;
  git_user_name?: string;
  git_user_email?: string;
  v1_enabled?: boolean;
  agent_settings_schema?: SettingsSchema | null;
  agent_settings?: Record<string, SettingsValue> | null;
  conversation_settings_schema?: SettingsSchema | null;
  conversation_settings?: Record<string, SettingsValue> | null;
  sandbox_grouping_strategy?: SandboxGroupingStrategy;
};
