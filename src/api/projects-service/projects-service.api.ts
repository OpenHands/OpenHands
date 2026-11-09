/**
 * ProjectsService reads the agent-server's /api/projects list, which names the
 * projects that conversations belong to (a conversation's ``project`` tag
 * holds its project's id). Agent servers without Projects answer 404.
 */
import { AgentServerClient } from "@openhands/typescript-client/clients";

import { getAgentServerClientOptions } from "../agent-server-client-options";

export interface ProjectSummary {
  id: string;
  name: string;
}

class ProjectsService {
  static async listProjects(): Promise<ProjectSummary[]> {
    const response = await new AgentServerClient(
      getAgentServerClientOptions(),
    ).request<{ projects?: Array<{ id?: unknown; name?: unknown }> }>({
      method: "GET",
      path: "/api/projects",
    });
    return (response?.projects ?? []).flatMap((project) =>
      typeof project.id === "string" && typeof project.name === "string"
        ? [{ id: project.id, name: project.name }]
        : [],
    );
  }
}

export default ProjectsService;
