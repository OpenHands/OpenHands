export const MCP_SERVER_NAME_PATTERN = /^[a-zA-Z0-9_-]+$/;

export function isValidMcpServerName(name: string): boolean {
  return MCP_SERVER_NAME_PATTERN.test(name);
}

export function toMcpServerName(value: string, fallback = "mcp"): string {
  const normalized = value
    .trim()
    .replace(/[^a-zA-Z0-9_-]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");

  return normalized || fallback;
}
