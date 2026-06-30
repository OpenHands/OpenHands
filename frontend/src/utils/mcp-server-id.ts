export interface McpServerListEntry {
  id: string;
  type: "sse" | "stdio" | "shttp";
  name?: string;
  url?: string;
  api_key?: string;
  timeout?: number;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
}

/** Canonical MCP server id used by the backend (mcpServers key). */
export function getMcpServerId(server: McpServerListEntry): string {
  if (server.type === "stdio") {
    return server.name || server.id;
  }
  if (server.name) {
    return server.name;
  }
  if (server.url) {
    return server.url;
  }
  return server.id;
}
