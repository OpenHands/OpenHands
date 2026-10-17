import { FileClient } from "@openhands/typescript-client/clients";
import type { AgentServerClientOptions } from "#/api/agent-server-client-options";

// @spec FD-001 — Preserve runtime bytes and release the client on every outcome
export async function downloadRuntimeFile(
  options: AgentServerClientOptions,
  path: string,
): Promise<ArrayBuffer> {
  const client = new FileClient(options);
  try {
    return await client.downloadFile(path);
  } finally {
    client.close();
  }
}
