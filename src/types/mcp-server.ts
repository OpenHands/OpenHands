import type { MCPTestFailureKind } from "@openhands/typescript-client";
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
}

export interface MCPTestToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

export interface MCPTestToolResult {
  is_error: boolean;
  text: string;
}

export type ExtendedMCPTestFailureKind = MCPTestFailureKind | "credentials";

export interface ExtendedMCPTestSuccess {
  ok: true;
  tools: string[];
  tool_result?: MCPTestToolResult | null;
  oauth_state?: MCPOAuthState | null;
}

export interface ExtendedMCPTestFailure {
  ok: false;
  error: string;
  error_kind: ExtendedMCPTestFailureKind;
}

export type ExtendedMCPTestResponse =
  | ExtendedMCPTestSuccess
  | ExtendedMCPTestFailure;

export interface MCPOAuthStartResponse {
  ok: boolean;
  job_id?: string | null;
  authorization_url?: string | null;
  error?: string | null;
  error_kind?: MCPTestFailureKind | null;
}

export interface MCPOAuthStatusResponse {
  ok: boolean;
  status: "pending" | "authorizing" | "succeeded" | "failed";
  job_id: string;
  authorization_url?: string | null;
  callback_ready?: boolean;
  tools?: string[] | null;
  tool_result?: MCPTestToolResult | null;
  oauth_state?: MCPOAuthState | null;
  error?: string | null;
  error_kind?: MCPTestFailureKind | null;
}
