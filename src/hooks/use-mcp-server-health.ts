import React from "react";
import {
  getMcpHealthSnapshot,
  subscribeMcpHealth,
} from "#/api/mcp-health/mcp-health-store";
import {
  probeMcpServerHealth,
  reauthorizeMcpServerHealth,
} from "#/api/mcp-health/probe-mcp-server-health";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { UNCHECKED_MCP_HEALTH, type McpServerHealth } from "#/types/mcp-health";
import type { MCPServerConfig } from "#/types/mcp-server";
import { getMcpServerHealthKey } from "#/utils/mcp-server-health-key";

/**
 * Subscribe to an installed server's connection health and expose the
 * probe actions bound to its current config.
 *
 * Health is keyed by both the active backend identity and the server's
 * structure: a single install of one catalog entry across two backends,
 * or across the same backend before and after a credential rotation, is
 * two independent health verdicts. See `McpServerHealthScope` and the
 * analogous backend-scoping fixes for plugins (#16843) and skills (#16893).
 */
export function useMcpServerHealth(server: MCPServerConfig) {
  const active = useActiveBackend();
  const key = getMcpServerHealthKey(
    {
      backendId: active.backend.id,
      connectionRevision: active.backend.connectionRevision,
    },
    server,
  );
  const health: McpServerHealth = React.useSyncExternalStore(
    subscribeMcpHealth,
    () => getMcpHealthSnapshot()[key] ?? UNCHECKED_MCP_HEALTH,
  );

  // Keep the latest config without re-creating the callbacks every render
  // (settings refetches rebuild the server objects each time).
  const serverRef = React.useRef(server);
  serverRef.current = server;
  const scopeRef = React.useRef({
    backendId: active.backend.id,
    connectionRevision: active.backend.connectionRevision,
  });
  scopeRef.current = {
    backendId: active.backend.id,
    connectionRevision: active.backend.connectionRevision,
  };

  const probe = React.useCallback(
    () => probeMcpServerHealth(scopeRef.current, serverRef.current),
    [],
  );
  const reauthorize = React.useCallback(
    () => reauthorizeMcpServerHealth(scopeRef.current, serverRef.current),
    [],
  );

  return { health, probe, reauthorize };
}
