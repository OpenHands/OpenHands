export type MCPOAuthClientAuthMethod =
  | "none"
  | "client_secret_post"
  | "client_secret_basic"
  | "private_key_jwt";

export type MCPJsonValue =
  | boolean
  | number
  | string
  | null
  | MCPJsonValue[]
  | { [key: string]: MCPJsonValue };

export interface MCPOAuthAuthenticationConfig {
  type: "oauth";
  client_auth_method?: MCPOAuthClientAuthMethod | null;
  scopes?: string | string[] | null;
  client_name?: string | null;
  client_metadata_url?: string | null;
  client_id?: string | null;
  client_secret?: string | null;
  additional_client_metadata?: Record<string, MCPJsonValue> | null;
}

export type MCPAuthenticationConfig = MCPOAuthAuthenticationConfig;

export interface MCPOAuthState {
  tokens?: Record<string, MCPJsonValue> | null;
  client_info?: Record<string, MCPJsonValue> | null;
  token_expires_at?: number | null;
}

export type MCPAuthCredential =
  | { strategy: "none" }
  | { strategy: "api_key"; value?: string | null; header_name?: string | null }
  | { strategy: "bearer"; value?: string | null }
  | { strategy: "basic"; username: string; password?: string | null }
  | { strategy: "header"; headers?: Record<string, string> | null }
  | {
      strategy: "oauth2";
      authentication?: MCPAuthenticationConfig | null;
      state?: MCPOAuthState | null;
    };

export const MCP_AUTH_STRATEGIES = [
  "none",
  "api_key",
  "bearer",
  "basic",
  "header",
  "oauth2",
] as const;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === "object" && !Array.isArray(value);

export const isMcpAuthCredential = (
  value: unknown,
): value is MCPAuthCredential =>
  isRecord(value) &&
  typeof value.strategy === "string" &&
  (MCP_AUTH_STRATEGIES as readonly string[]).includes(value.strategy);
