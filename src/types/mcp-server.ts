import type {
  AgentServerMCPOAuthStatusResponse,
  AgentServerMCPStartOAuthResponse,
  AgentServerMCPTestResponse,
  AgentServerMCPToolCall,
  AgentServerMCPToolCallResult,
  MCPTestFailureKind,
} from "@openhands/typescript-client";
import type { MCPAuthCredential, MCPOAuthState } from "./mcp-auth";

export type MCPServerType = "sse" | "stdio" | "shttp";

export interface MCPServerConfig {
  id: string;
  type: MCPServerType;
  name?: string;
  url?: string;
  headers?: Record<string, string>;
  timeout?: number;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  auth?: MCPAuthCredential;
  /**
   * Whether the agent may use this server. Absent means enabled — only an
   * explicit `false` disables, so existing configs stay untouched.
   */
  enabled?: boolean;
}

export type MCPTestToolCall = AgentServerMCPToolCall;
export type MCPTestToolResult = AgentServerMCPToolCallResult;

export type ExtendedMCPTestFailureKind = MCPTestFailureKind | "credentials";

type GeneratedMCPTestSuccess = Extract<
  AgentServerMCPTestResponse,
  { ok?: true }
>;
type GeneratedMCPTestFailure = Extract<
  AgentServerMCPTestResponse,
  { error: string }
>;

export type ExtendedMCPTestSuccess = Omit<
  GeneratedMCPTestSuccess,
  "ok" | "tools" | "oauth_state"
> & {
  ok: true;
  tools: string[];
  tool_result?: MCPTestToolResult | null;
  oauth_state?: MCPOAuthState | null;
};

export type ExtendedMCPTestFailure = Omit<
  GeneratedMCPTestFailure,
  "ok" | "error_kind"
> & {
  ok: false;
  error_kind: ExtendedMCPTestFailureKind;
};

export type ExtendedMCPTestResponse =
  | ExtendedMCPTestSuccess
  | ExtendedMCPTestFailure;

export type MCPOAuthStartResponse = AgentServerMCPStartOAuthResponse;
export type MCPOAuthStatusResponse = Omit<
  AgentServerMCPOAuthStatusResponse,
  "oauth_state"
> & {
  oauth_state?: MCPOAuthState | null;
};
