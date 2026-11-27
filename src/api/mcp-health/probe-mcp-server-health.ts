import McpService from "#/api/mcp-service/mcp-service.api";
import type { McpServerHealth } from "#/types/mcp-health";
import type {
  ExtendedMCPTestResponse,
  MCPServerConfig,
} from "#/types/mcp-server";
import { getCredentialValidationForServer } from "#/utils/mcp-credential-validation";
import { getMcpServerHealthKey } from "#/utils/mcp-server-health-key";
import { redactMcpSecrets } from "#/utils/redact-mcp-secrets";
import {
  beginMcpHealthCheck,
  resolveMcpHealthCheck,
  setMcpServerHealth,
} from "./mcp-health-store";

/**
 * Conservative auth-failure sniffing for hosted servers that reject a bad
 * token at the HTTP handshake: the backend can only classify those as
 * `connection`/`unknown` (the MCP connection itself failed), but the
 * guidance the user needs is the credentials one.
 */
const AUTH_FAILURE_TEXT =
  /\b(401|403)\b|unauthorized|forbidden|invalid[ _-]?(token|credentials?|api[ _-]?key)/i;

function failedHealth(
  server: MCPServerConfig,
  error: unknown,
): McpServerHealth {
  const message = error instanceof Error ? error.message : String(error);
  return {
    status: "failed",
    kind: "unknown",
    error: redactMcpSecrets(message, server),
    checkedAt: Date.now(),
  };
}

/**
 * Map a test/OAuth probe response (already secret-redacted and
 * credential-interpreted by `McpService`) to a card health state.
 *
 * `verified` requires the catalog entry's read-only probe tool to have been
 * advertised AND invoked without error; everything else that connected is
 * `connectivity-only` — truthfully labeled, never silently upgraded.
 */
export function interpretMcpTestResponse(
  server: MCPServerConfig,
  response: ExtendedMCPTestResponse,
): McpServerHealth {
  const checkedAt = Date.now();
  if (!response.ok) {
    const kind =
      (response.error_kind === "connection" ||
        response.error_kind === "unknown") &&
      AUTH_FAILURE_TEXT.test(response.error)
        ? "credentials"
        : response.error_kind;
    return { status: "failed", kind, error: response.error, checkedAt };
  }
  const validation = getCredentialValidationForServer(server);
  const verified =
    !!validation &&
    response.tools.includes(validation.toolCall.name) &&
    !!response.tool_result &&
    !response.tool_result.is_error;
  return {
    status: "healthy",
    verification: verified ? "verified" : "connectivity-only",
    toolCount: response.tools.length,
    checkedAt,
  };
}

/** Run the non-mutating connection probe and publish the result. */
export async function probeMcpServerHealth(
  server: MCPServerConfig,
): Promise<void> {
  const key = getMcpServerHealthKey(server);
  const checkId = beginMcpHealthCheck(key);
  let health: McpServerHealth;
  try {
    const response = await McpService.testServer(server);
    health = interpretMcpTestResponse(server, response);
  } catch (error) {
    health = failedHealth(server, error);
  }
  resolveMcpHealthCheck(key, checkId, health);
}

/**
 * Run the interactive OAuth authorization probe and publish the result.
 * Returns the raw response (null on transport error) so the caller can
 * persist a refreshed `oauth_state`.
 */
export async function reauthorizeMcpServerHealth(
  server: MCPServerConfig,
): Promise<ExtendedMCPTestResponse | null> {
  const key = getMcpServerHealthKey(server);
  const checkId = beginMcpHealthCheck(key);
  try {
    const response = await McpService.authorizeOAuth(server);
    resolveMcpHealthCheck(
      key,
      checkId,
      interpretMcpTestResponse(server, response),
    );
    return response;
  } catch (error) {
    resolveMcpHealthCheck(key, checkId, failedHealth(server, error));
    return null;
  }
}

/**
 * Publish a just-saved server's health from the pre-save test result the
 * install/edit flow already produced (same config, moments earlier), so its
 * card shows a verdict immediately without a second probe.
 *
 * `otherServers` are the servers whose health must NOT be overwritten —
 * callers pass the pre-save list, excluding the server being saved. When a
 * same-shape duplicate exists (health keys exclude secret values, so two
 * installs of one catalog entry can collide until the save suffixes the
 * name), the seed is skipped and the new card simply starts "unchecked".
 */
export function seedMcpServerHealth(
  server: MCPServerConfig,
  response: ExtendedMCPTestResponse,
  otherServers: MCPServerConfig[],
): void {
  const key = getMcpServerHealthKey(server);
  if (otherServers.some((other) => getMcpServerHealthKey(other) === key)) {
    return;
  }
  setMcpServerHealth(key, interpretMcpTestResponse(server, response));
}
