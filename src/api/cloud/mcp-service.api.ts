import type { AgentServerMCPTestRequest } from "@openhands/typescript-client";
import { getActiveBackend } from "../backend-registry/active-store";
import type { Backend } from "../backend-registry/types";
import type { ExtendedMCPTestResponse } from "#/types/mcp-server";
import { callCloudProxy } from "./proxy";

const DEFAULT_MCP_TEST_TIMEOUT_SECONDS = 15;

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
