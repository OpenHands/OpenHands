export type McpServerTestRunStatus =
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled";

export type McpServerHealthStatus =
  | "unknown"
  | "testing"
  | "healthy"
  | "unhealthy";

export type McpServerFailureCategory =
  | "configuration"
  | "connection"
  | "authentication"
  | "protocol"
  | "tool_discovery"
  | "execution"
  | "sandbox"
  | "timeout"
  | "internal";

export interface McpServerTestRun {
  id: string;
  created_by_user_id?: string | null;
  server_id: string;
  transport: "stdio" | "sse" | "shttp";
  status: McpServerTestRunStatus;
  category?: McpServerFailureCategory | null;
  message?: string | null;
  tool_count?: number | null;
  latency_ms?: number | null;
  sandbox_id?: string | null;
  started_at: string;
  finished_at?: string | null;
  created_at: string;
}

export interface McpServerTestRunPage {
  items: McpServerTestRun[];
  next_page_id?: string | null;
}

export interface StartMcpServerTestResponse {
  test_id: string;
  status: McpServerTestRunStatus;
}

export interface McpServerHealthResponse {
  server_id: string;
  status: McpServerHealthStatus;
  category?: McpServerFailureCategory | null;
  message?: string | null;
  tool_count?: number | null;
  latency_ms?: number | null;
  tested_at?: string | null;
  test_id?: string | null;
}
