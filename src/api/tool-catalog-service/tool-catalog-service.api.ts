import { ToolClient } from "@openhands/typescript-client/clients";
import type { ToolCatalogEntry } from "@openhands/typescript-client";

import { getAgentServerClientOptions } from "../agent-server-client-options";

export type { ToolCatalogEntry };

function isToolCatalogEntry(value: unknown): value is ToolCatalogEntry {
  if (typeof value !== "object" || value === null) return false;
  const entry = value as Record<string, unknown>;
  return (
    typeof entry.name === "string" &&
    typeof entry.user_selectable === "boolean" &&
    typeof entry.usable === "boolean" &&
    typeof entry.in_default_set === "boolean" &&
    typeof entry.description === "string"
  );
}

function assertValidCatalog(
  tools: unknown[],
): asserts tools is ToolCatalogEntry[] {
  for (const tool of tools) {
    if (!isToolCatalogEntry(tool)) {
      const name =
        typeof tool === "object" && tool !== null
          ? ((tool as Record<string, unknown>).name ?? "<missing>")
          : "<malformed>";
      throw new Error(
        `The agent server returned a malformed tool catalog entry ` +
          `(name="${String(name)}"); refusing to pick tools from it.`,
      );
    }
  }
}

class ToolCatalogService {
  /** Tools this server offers, in its order. */
  static async getCatalog(): Promise<ToolCatalogEntry[]> {
    const response = await new ToolClient(
      getAgentServerClientOptions(),
    ).getToolCatalog();
    const tools: unknown = response?.tools;
    if (!Array.isArray(tools)) {
      throw new Error(
        "The agent server returned a malformed tool catalog (missing the tools array).",
      );
    }
    assertValidCatalog(tools);
    return tools;
  }
}

export default ToolCatalogService;
