import { Provider } from "#/types/settings";

export type ApiSettings = {
  llm_model: string;
  llm_base_url: string;
  agent: string;
  language: string;
  llm_api_key: string | null;
  llm_api_key_set: boolean;
  search_api_key_set: boolean;
  confirmation_mode: boolean;
  security_analyzer: string | null;
  remote_runtime_resource_factor: number | null;
  enable_default_condenser: boolean;
  // Max size for condenser in backend settings
  condenser_max_size: number | null;
  enable_sound_notifications: boolean;
  enable_proactive_conversation_starters: boolean;
  enable_solvability_analysis: boolean;
  user_consents_to_analytics: boolean | null;
  search_api_key?: string;
  provider_tokens_set: Partial<Record<Provider, string | null>>;
  max_budget_per_task: number | null;
  mcp_config?: {
    sse_servers: (string | { url: string; api_key?: string })[];
    stdio_servers: {
      name: string;
      command: string;
      args?: string[];
      env?: Record<string, string>;
    }[];
    shttp_servers: (string | { url: string; api_key?: string })[];
  };
  email?: string;
  email_verified?: boolean;
  git_user_name?: string;
  git_user_email?: string;
  v1_enabled?: boolean;
};

export type PostApiSettings = ApiSettings & {
  user_consents_to_analytics: boolean | null;
  search_api_key?: string;
  mcp_config?: {
    sse_servers: (string | { url: string; api_key?: string })[];
    stdio_servers: {
      name: string;
      command: string;
      args?: string[];
      env?: Record<string, string>;
    }[];
    shttp_servers: (string | { url: string; api_key?: string })[];
  };
};
