import { AgentServerClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "./agent-server-client-options";

// Temporary mirror of software-agent-sdk@5322eb5 until changeConfiguration is published.
export type ConversationConfigurationOperation =
  | { type: "secret.grant"; name: string; value: string }
  | { type: "mcp.attach"; ref: string };

export interface ConversationConfigurationChangesRequest {
  operations: [ConversationConfigurationOperation];
}

export interface ConversationConfigurationChangesResponse {
  applied: [
    | { type: "secret.grant"; name: string }
    | { type: "mcp.attach"; ref: string },
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
