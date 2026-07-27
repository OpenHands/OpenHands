import React from "react";
import {
  getMcpHealthSnapshot,
  subscribeMcpHealth,
} from "#/api/mcp-health/mcp-health-store";
import {
  probeMcpServerHealth,
  reauthorizeMcpServerHealth,
} from "#/api/mcp-health/probe-mcp-server-health";
import { UNCHECKED_MCP_HEALTH, type McpServerHealth } from "#/types/mcp-health";
import type { MCPServerConfig } from "#/types/mcp-server";
import { getMcpServerHealthKey } from "#/utils/mcp-server-health-key";

/**
 * Subscribe to an installed server's connection health and expose the
 * probe actions bound to its current config.
 */
export function useMcpServerHealth(server: MCPServerConfig) {
  const key = getMcpServerHealthKey(server);
  const health: McpServerHealth = React.useSyncExternalStore(
    subscribeMcpHealth,
    () => getMcpHealthSnapshot()[key] ?? UNCHECKED_MCP_HEALTH,
  );

  // Keep the latest config without re-creating the callbacks every render
  // (settings refetches rebuild the server objects each time).
  const serverRef = React.useRef(server);
  serverRef.current = server;

  const probe = React.useCallback(
    () => probeMcpServerHealth(serverRef.current),
    [],
  );
  const reauthorize = React.useCallback(
    () => reauthorizeMcpServerHealth(serverRef.current),
    [],
  );

  return { health, probe, reauthorize };
}
