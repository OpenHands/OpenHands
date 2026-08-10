import type {
  MCPAuthCredential,
  MCPOAuthAuthentication,
} from "@openhands/typescript-client";

export type {
  MCPAuthCredential,
  MCPOAuthAuthentication,
  MCPOAuthState,
} from "@openhands/typescript-client";

export type MCPAuthenticationConfig = MCPOAuthAuthentication;
export type MCPOAuthAuthenticationConfig = MCPOAuthAuthentication;
export type MCPOAuthClientAuthMethod = NonNullable<
  MCPOAuthAuthentication["client_auth_method"]
>;

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
