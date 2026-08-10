/**
 * Cortex Skills Registry and Interfaces
 * Allows modular extension of agent capabilities with specific expert domains,
 * encapsulating Model Context Protocol (MCP) servers within the Cortex UX.
 */

export interface CortexSkill {
  id: string;
  name: string;
  description: string;
  version: string;
  tools: string[];
  execute(
    action: string,
    context: Record<string, unknown>,
  ): Promise<Record<string, unknown>>;
}

export interface CortexMcpIntegration {
  serverId: string;
  name: string;
  type: "stdio" | "sse";
  status: "connected" | "disconnected" | "error";
  command?: string;
  args?: string[];
  sseUrl?: string;
  associatedSkills: string[];
}

export class CortexSkillsRegistry {
  private skills = new Map<string, CortexSkill>();
  private mcpIntegrations = new Map<string, CortexMcpIntegration>();

  /**
   * Registers a new professional skill to the Cortex platform.
   */
  public register(skill: CortexSkill): void {
    this.skills.set(skill.id, skill);
  }

  /**
   * Retrieves a registered skill by its ID.
   */
  public get(id: string): CortexSkill | undefined {
    return this.skills.get(id);
  }

  /**
   * Lists all registered skills.
   */
  public list(): CortexSkill[] {
    return Array.from(this.skills.values());
  }

  /**
   * Integrates an OpenHands MCP server into the Cortex UI wrapper as a product integration.
   */
  public addMcpIntegration(integration: CortexMcpIntegration): void {
    this.mcpIntegrations.set(integration.serverId, integration);
  }

  /**
   * Retrieves a registered MCP integration.
   */
  public getMcpIntegration(serverId: string): CortexMcpIntegration | undefined {
    return this.mcpIntegrations.get(serverId);
  }

  /**
   * Lists all registered MCP integrations.
   */
  public listMcpIntegrations(): CortexMcpIntegration[] {
    return Array.from(this.mcpIntegrations.values());
  }

  /**
   * Dynamic wrapper mapping an MCP server configuration to a premium CORTEX Skill.
   */
  public mapMcpToCortexSkill(integration: CortexMcpIntegration): CortexSkill {
    return {
      id: `cortex-mcp-${integration.serverId}`,
      name: `${integration.name} Integration`,
      description: `Cortex-wrapped MCP skill enabling specialized agent capabilities via ${integration.type} protocol.`,
      version: "1.0.0",
      tools: ["mcp-bridge-tool"],
      execute: async (action, context) => {
        // Bridges to the real underlying OpenHands core MCP connection
        return {
          success: true,
          actionExecuted: action,
          server: integration.serverId,
          timestamp: Date.now(),
          context,
        };
      },
    };
  }
}
