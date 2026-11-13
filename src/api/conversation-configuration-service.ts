import type { MCPConfig, MCPServer } from "@openhands/typescript-client";
import { AgentServerClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "./agent-server-client-options";

// Temporary mirrors of software-agent-sdk endpoints until its TypeScript client is published.
export interface AddConversationMcpServersRequest {
  mcp_servers: MCPConfig;
  save_to_profile?: boolean;
}

export interface UpdateConversationSecretsRequest {
  secrets: Record<string, string>;
  save_to_profile?: boolean;
}

export interface ConversationConfigurationSuccess {
  success: boolean;
}

export function selectConversationMcpServers(
  mcpConfig: MCPConfig | undefined,
  refs: string[],
): MCPConfig {
  const selected: MCPConfig = {};
  for (const ref of refs) {
    const server = mcpConfig?.[ref] as MCPServer | undefined;
    if (server) selected[ref] = server;
  }
  return selected;
}

export async function addConversationMcpServers(
  conversationId: string,
  request: AddConversationMcpServersRequest,
): Promise<ConversationConfigurationSuccess> {
  const client = new AgentServerClient(getAgentServerClientOptions());
  try {
    return await client.post<ConversationConfigurationSuccess>(
      `/api/conversations/${encodeURIComponent(conversationId)}/mcp`,
      request,
    );
  } finally {
    client.close();
  }
}

export async function updateConversationSecrets(
  conversationId: string,
  request: UpdateConversationSecretsRequest,
): Promise<ConversationConfigurationSuccess> {
  const client = new AgentServerClient(getAgentServerClientOptions());
  try {
    return await client.post<ConversationConfigurationSuccess>(
      `/api/conversations/${encodeURIComponent(conversationId)}/secrets`,
      request,
    );
  } finally {
    client.close();
  }
}
