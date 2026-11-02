import type { AgentServerMCPTestRequest } from "@openhands/typescript-client";
import { getActiveBackend } from "../backend-registry/active-store";
import type { Backend } from "../backend-registry/types";
import type {
  ExtendedMCPTestResponse,
  MCPOAuthStartResponse,
  MCPOAuthStatusResponse,
} from "#/types/mcp-server";
import { callCloudProxy } from "./proxy";

const DEFAULT_MCP_TEST_TIMEOUT_SECONDS = 15;
// `POST /api/v1/mcp/oauth/start` answers as soon as the provider's
// authorization URL is known and waits at most 30s for it.
const OAUTH_START_DEADLINE_SECONDS = 35;

function getActiveCloudBackend(): Backend {
  const active = getActiveBackend().backend;
  if (active.kind !== "cloud") {
    throw new Error("Cloud MCP test call requires a cloud backend.");
  }
  return active;
}

/**
 * Probe a remote MCP server through the cloud backend's
 * `POST /api/v1/mcp/test`, which shares the agent-server test contract
 * (HTTP 200 with `ok: false` for connection/timeout failures). The probe
 * lists tools and may then run one read-only tool call, each bounded by
 * `timeout`, so the request deadline covers both plus a margin.
 */
export async function testCloudMcpServer(
  request: AgentServerMCPTestRequest,
): Promise<ExtendedMCPTestResponse> {
  const backend = getActiveCloudBackend();
  const timeout = request.timeout ?? DEFAULT_MCP_TEST_TIMEOUT_SECONDS;
  return callCloudProxy<ExtendedMCPTestResponse>({
    backend,
    method: "POST",
    path: "/api/v1/mcp/test",
    body: request,
    timeoutSeconds: 2 * timeout + 5,
  });
}

/**
 * Start an OAuth install probe through the cloud backend's
 * `POST /api/v1/mcp/oauth/start`. Same contract as the agent-server route:
 * the response carries the provider's authorization URL for the popup, the
 * provider redirects that popup to the app server's own callback, and
 * `getCloudMcpOAuthStatus` reports progress and the final OAuth state.
 */
export async function startCloudMcpOAuth(
  request: AgentServerMCPTestRequest,
): Promise<MCPOAuthStartResponse> {
  const backend = getActiveCloudBackend();
  return callCloudProxy<MCPOAuthStartResponse>({
    backend,
    method: "POST",
    path: "/api/v1/mcp/oauth/start",
    body: request,
    timeoutSeconds: OAUTH_START_DEADLINE_SECONDS,
  });
}

export async function getCloudMcpOAuthStatus(
  jobId: string,
): Promise<MCPOAuthStatusResponse> {
  const backend = getActiveCloudBackend();
  return callCloudProxy<MCPOAuthStatusResponse>({
    backend,
    method: "GET",
    path: `/api/v1/mcp/oauth/status/${encodeURIComponent(jobId)}`,
  });
}
