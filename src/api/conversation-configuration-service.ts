import type { MCPServer } from "@openhands/typescript-client";
import { AgentServerClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "./agent-server-client-options";

// Temporary mirror of software-agent-sdk#6710 until its TypeScript client is published.
export type ConversationConfigurationOperation =
  | { type: "secret.grant"; name: string; value: string }
  | { type: "mcp.attach"; ref: string }
  | { type: "mcp.install"; name: string; server: MCPServer };

export interface ConversationConfigurationChangesRequest {
  operations: [ConversationConfigurationOperation];
}

export interface ConversationConfigurationChangesResponse {
  applied: [
    | { type: "secret.grant"; name: string }
    | { type: "mcp.attach"; ref: string }
    | { type: "mcp.install"; name: string },
  ];
}

export async function changeConversationConfiguration(
  conversationId: string,
  request: ConversationConfigurationChangesRequest,
): Promise<ConversationConfigurationChangesResponse> {
  const client = new AgentServerClient(getAgentServerClientOptions());
  try {
    return await client.post<ConversationConfigurationChangesResponse>(
      `/api/conversations/${encodeURIComponent(conversationId)}/configuration/changes`,
      request,
    );
  } finally {
    client.close();
  }
}
