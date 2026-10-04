import type { MCPConfig } from "@openhands/typescript-client";
import type { MCPServerConfig } from "#/types/mcp-server";
import { getMcpServerEnabled } from "./mcp-config";

export function flattenMcpConfig(config: MCPConfig): MCPServerConfig[] {
  return Object.entries(config).map(([settingsKey, server]) =>
    server.transport === "stdio"
      ? {
          id: settingsKey,
          type: "stdio",
          name: settingsKey,
          command: server.command,
          args: server.args ?? undefined,
          env: server.env ?? undefined,
          enabled: getMcpServerEnabled(server),
        }
      : {
          id: settingsKey,
          type: server.transport === "sse" ? "sse" : "shttp",
          name: settingsKey,
          url: server.url,
          headers: server.headers ?? undefined,
          timeout: server.timeout ?? undefined,
          auth: server.auth ?? undefined,
          enabled: getMcpServerEnabled(server),
        },
  );
}
