import { SubAgentsClient } from "@openhands/typescript-client/clients";
import type { AgentInfo } from "#/types/settings";
import { getAgentServerWorkingDir } from "./agent-server-config";
import { getActiveBackend } from "./backend-registry/active-store";
import { fetchCloudAgents } from "./cloud/agents-service.api";
import { getAgentServerClientOptions } from "./agent-server-client-options";

class AgentsService {
  /**
   * List the file-based and built-in sub-agents available to a workspace.
   *
   * Unlike skills, agents have no bundled public catalog: the agent-server
   * discovers project (`{workspace}/.agents/agents`), user (`~/.agents/agents`)
   * and SDK built-in agents itself, so the listing is whatever
   * `POST /api/sub-agents` returns.
   */
  static async getAgents(projectDir?: string): Promise<AgentInfo[]> {
    if (getActiveBackend().backend.kind === "cloud") {
      return fetchCloudAgents();
    }

    try {
      const response = await new SubAgentsClient(
        getAgentServerClientOptions(),
      ).getSubAgents({
        load_user: true,
        load_project: true,
        load_builtin: true,
        project_dir: projectDir ?? getAgentServerWorkingDir(),
      });
      return (response.agents ?? []) as AgentInfo[];
    } catch {
      // Older agent-servers don't expose the sub-agents endpoint, and a local
      // server may be unreachable; degrade to an empty catalog rather than
      // breaking the Sub-Agents page.
      return [];
    }
  }
}

export default AgentsService;
