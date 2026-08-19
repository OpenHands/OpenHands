import { useQuery } from "@tanstack/react-query";
import AgentsService from "#/api/agents-service";
import { AgentInfo } from "#/types/settings";

/**
 * @param projectDir Workspace root to load project agents from. Conversation
 *   views pass the conversation's own workspace so the catalog matches the
 *   agents loaded into that conversation; the global Agents page omits it.
 */
export const useAgents = (projectDir?: string) =>
  useQuery<AgentInfo[]>({
    queryKey: ["agents", projectDir ?? null],
    queryFn: () => AgentsService.getAgents(projectDir),
    staleTime: 1000 * 60 * 10, // 10 minutes – agent list rarely changes
    refetchOnWindowFocus: false,
  });
